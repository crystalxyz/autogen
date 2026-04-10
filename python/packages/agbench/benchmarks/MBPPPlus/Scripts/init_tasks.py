#
# Run this file to download MBPP+ (the EvalPlus-augmented MBPP) and create
# the corresponding agbench scenario JSONL files (one per template).
#
# MBPP+ is the augmented MBPP from the EvalPlus suite. As with HumanEvalPlus,
# we use the canonical solution as an oracle to compute expected outputs over
# `base_input + plus_input`, and synthesize a `check(candidate)` function the
# AgentChat template can drop into its standard test harness.
#
# Two data sources are supported (in order of preference):
#   1. The `evalplus` Python package (`pip install evalplus`) — canonical.
#   2. The HuggingFace `evalplus/mbppplus` dataset, if it exposes a
#      ready-made `test` field.
#
# Note: MBPP problems are typically described in natural language, not as a
# stub function. The original MBPP `prompt` field already contains the
# function signature plus a docstring describing the task and showing one or
# more example assertions, which is what the agent receives verbatim.
#

import base64
import json
import os
import pickle
import re

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")


def _build_check_from_oracle(prompt: str, canonical_solution: str, entry_point: str,
                             base_input, plus_input, atol) -> str:
    """Build a `check(candidate)` function that, at test time, runs the
    canonical solution as an oracle alongside the candidate and compares
    outputs per case. Embedding the canonical solution (instead of
    materializing expected outputs at init time) avoids exploding the
    on-disk JSONL when EvalPlus stress tests produce gigabyte-scale
    expected values (e.g. `combinations_with_replacement` with n=77)."""
    inputs = list(base_input or []) + list(plus_input or [])
    # Drop inputs that aren't picklable (e.g. ones containing re.Match).
    safe_inputs = []
    for args in inputs:
        try:
            pickle.dumps(args)
        except Exception:
            continue
        safe_inputs.append(args)

    inputs_blob = base64.b64encode(pickle.dumps(safe_inputs)).decode("ascii")
    canonical_blob = base64.b64encode(canonical_solution.encode("utf-8")).decode("ascii")
    prompt_blob = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
    use_atol = bool(atol)
    atol_repr = repr(atol) if use_atol else "0"
    return f'''

def check(candidate):
    import base64, pickle, math
    _INPUTS = pickle.loads(base64.b64decode("{inputs_blob}"))
    _CANON_SRC = base64.b64decode("{canonical_blob}").decode("utf-8")
    _PROMPT_SRC = base64.b64decode("{prompt_blob}").decode("utf-8")
    _ENTRY = {entry_point!r}
    _ATOL = {atol_repr}

    _ns = {{}}
    try:
        exec(_CANON_SRC, _ns)
        if _ENTRY not in _ns:
            raise NameError(_ENTRY)
    except Exception:
        _ns = {{}}
        exec(_PROMPT_SRC + "\\n" + _CANON_SRC, _ns)
    _oracle = _ns[_ENTRY]

    def _close(a, b):
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
        if isinstance(a, float) or isinstance(b, float):
            try:
                return math.isclose(a, b, rel_tol=max(_ATOL, 1e-9), abs_tol=max(_ATOL, 1e-9))
            except Exception:
                return a == b
        return a == b

    for _args in _INPUTS:
        try:
            _exp = _oracle(*_args)
        except Exception:
            # Oracle itself can't handle this input -- skip it.
            continue
        _out = candidate(*_args)
        if _ATOL:
            assert _close(_out, _exp), "case failed (atol)"
        else:
            assert _out == _exp, "case failed"
'''


def _load_via_evalplus():
    from evalplus.data import get_mbpp_plus  # type: ignore

    raw = get_mbpp_plus()
    tasks = []
    for task_id, p in raw.items():
        prompt = p["prompt"]
        entry_point = p["entry_point"]
        canonical_solution = p["canonical_solution"]
        base_input = p.get("base_input") or []
        plus_input = p.get("plus_input") or []
        atol = p.get("atol", 0)
        test = _build_check_from_oracle(
            prompt, canonical_solution, entry_point, base_input, plus_input, atol
        )
        tasks.append({
            "task_id": task_id,
            "prompt": prompt,
            "test": test,
            "entry_point": entry_point,
        })
    return tasks


def _load_via_huggingface():
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("evalplus/mbppplus", split="test")
    if "test" not in ds.column_names:
        raise RuntimeError(
            "evalplus/mbppplus does not expose a ready-made `test` field; "
            "install the `evalplus` package instead (pip install evalplus)."
        )
    tasks = []
    for row in ds:
        tasks.append({
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
        })
    return tasks


def download_mbpp_plus():
    try:
        return _load_via_evalplus()
    except ImportError:
        print("[info] `evalplus` package not installed; falling back to HuggingFace.")
        return _load_via_huggingface()


def create_jsonl(name, tasks, template):
    if not os.path.isdir(TASKS_DIR):
        os.makedirs(TASKS_DIR)

    out_path = os.path.join(TASKS_DIR, name + ".jsonl")
    with open(out_path, "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['task_id']}")
            record = {
                "id": str(task["task_id"]).replace("/", "_"),
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": task["prompt"]},
                    "test.txt": {"__TEST__": task["test"]},
                    "custom_code_executor.py": {"__ENTRY_POINT__": task["entry_point"]},
                },
            }
            fh.write(json.dumps(record).strip() + "\n")
    print(f"Wrote {out_path}")


def main():
    tasks = download_mbpp_plus()
    print(f"Loaded {len(tasks)} MBPP+ problems")

    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    for name, path in templates.items():
        create_jsonl(f"mbpp_plus_{name}", tasks, path)


if __name__ == "__main__" and __package__ is None:
    main()
