#
# Run this file to download the LiveCodeBench dataset and create the
# corresponding agbench scenario JSONL files (one per template).
#
# LiveCodeBench is hosted on the Hugging Face Hub. The default config below
# pulls the "code_generation_lite" subset, release_v1, which is the
# canonical small/fast subset used by most papers. You can override either
# of these with --release / --subset.
#

import argparse
import base64
import json
import os
import pickle
import re
import zlib

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")


def _decode_private_tests(raw):
    """LiveCodeBench encodes private_test_cases as base64(zlib(pickle([...])))
    when the JSON would otherwise be very large. Smaller problems just store
    the raw JSON list. Handle both."""
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        # First try plain JSON
        try:
            return json.loads(raw)
        except Exception:
            pass
        # Then try base64 -> zlib -> pickle (LCB lite encoding). The pickled
        # payload is itself a JSON string, so json.loads it back to a list.
        try:
            decoded = pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8"))))
            if isinstance(decoded, str):
                return json.loads(decoded)
            return decoded
        except Exception:
            pass
        # Last resort: base64 -> json
        try:
            return json.loads(base64.b64decode(raw.encode("utf-8")).decode("utf-8"))
        except Exception:
            return []
    return []


def _normalize_tests(tests):
    """Ensure tests are a list of dicts with input/output/testtype string fields."""
    norm = []
    for t in tests or []:
        if not isinstance(t, dict):
            continue
        norm.append({
            "input": t.get("input", ""),
            "output": t.get("output", ""),
            "testtype": t.get("testtype", "stdin"),
        })
    return norm


def _fn_name_from_metadata(meta_field):
    if not meta_field:
        return ""
    if isinstance(meta_field, dict):
        return meta_field.get("func_name", "") or ""
    if isinstance(meta_field, str):
        try:
            return (json.loads(meta_field) or {}).get("func_name", "") or ""
        except Exception:
            return ""
    return ""


RELEASE_TO_FILE = {
    "release_v1": "test.jsonl",
    "release_v2": "test2.jsonl",
    "release_v3": "test3.jsonl",
    "release_v4": "test4.jsonl",
    "release_v5": "test5.jsonl",
    "release_v6": "test6.jsonl",
}


def download_livecodebench(release: str, subset: str):
    """Load LiveCodeBench from Hugging Face by directly downloading the per-release
    jsonl file from the dataset repo (the repo's loader script is no longer
    supported by recent `datasets` versions)."""
    from huggingface_hub import hf_hub_download  # type: ignore

    fname = RELEASE_TO_FILE.get(release)
    if fname is None:
        raise ValueError(f"Unknown release {release!r}; known: {list(RELEASE_TO_FILE)}")

    local_path = hf_hub_download(
        repo_id=f"livecodebench/{subset}",
        filename=fname,
        repo_type="dataset",
    )

    rows = []
    with open(local_path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    out = []
    for row in rows:
        public_tests = _normalize_tests(_decode_private_tests(row.get("public_test_cases")))
        private_tests = _normalize_tests(_decode_private_tests(row.get("private_test_cases")))
        fn_name = _fn_name_from_metadata(row.get("metadata"))

        question_id = (
            row.get("question_id")
            or row.get("task_id")
            or row.get("question_title")
            or f"lcb_{len(out)}"
        )
        out.append({
            "question_id": str(question_id),
            "question_content": row.get("question_content") or row.get("question") or "",
            "starter_code": row.get("starter_code") or "",
            "public_tests": public_tests,
            "private_tests": private_tests,
            "fn_name": fn_name,
        })
    return out


def _safe_id(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", qid)


def create_jsonl(name, tasks, template):
    if not os.path.isdir(TASKS_DIR):
        os.makedirs(TASKS_DIR)

    out_path = os.path.join(TASKS_DIR, name + ".jsonl")
    with open(out_path, "wt") as fh:
        for task in tasks:
            qid = _safe_id(task["question_id"])
            print(f"Converting: [{name}] {qid}")
            record = {
                "id": qid,
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": task["question_content"]},
                    "starter_code.txt": {"__STARTER_CODE__": task["starter_code"]},
                    "public_tests.json": {"__PUBLIC_TESTS__": json.dumps(task["public_tests"])},
                    "private_tests.json": {"__PRIVATE_TESTS__": json.dumps(task["private_tests"])},
                    "meta.json": {
                        "__FN_NAME__": task["fn_name"],
                        "__QUESTION_ID__": task["question_id"],
                    },
                },
            }
            fh.write(json.dumps(record).strip() + "\n")
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Initialize LiveCodeBench tasks for agbench.")
    parser.add_argument("--release", default="release_v1",
                        help="LiveCodeBench release tag (default: release_v1)")
    parser.add_argument("--subset", default="code_generation_lite",
                        help="HF subset name under the livecodebench org "
                             "(default: code_generation_lite)")
    args = parser.parse_args()

    tasks = download_livecodebench(args.release, args.subset)
    print(f"Loaded {len(tasks)} LiveCodeBench problems "
          f"(subset={args.subset}, release={args.release})")

    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    for name, path in templates.items():
        create_jsonl(f"livecodebench_{name}", tasks, path)


if __name__ == "__main__" and __package__ is None:
    main()
