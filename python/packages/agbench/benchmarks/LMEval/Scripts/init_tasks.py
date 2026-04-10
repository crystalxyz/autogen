"""Generate LMEval Tasks JSONL files.

Each task record maps one lm-eval task name to the LMEvalRunner template.
Run this script to (re)create benchmarks/LMEval/Tasks/lm_eval_default.jsonl
and optional curated subsets.
"""

import json
import os
from typing import List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
# Use absolute paths (mirrors HumanEval's init_tasks.py) so `template`
# resolves regardless of the JSONL's location relative to Templates/.
TEMPLATE_DIRECT = os.path.join(SCENARIO_DIR, "Templates", "LMEvalRunner")
TEMPLATE_AGENT = os.path.join(SCENARIO_DIR, "Templates", "LMEvalAgent")


# (task_name, num_fewshot, limit)  --  limit=0 means "all samples"
DEFAULT_TASKS: List[Tuple[str, int, int]] = [
    ("gsm8k", 5, 0),
    ("triviaqa", 5, 0),
    ("drop", 3, 0),
    ("mmlu", 5, 0),
    ("arc_easy", 0, 0),
    ("arc_challenge", 0, 0),
    ("hellaswag", 0, 0),
    ("truthfulqa_mc2", 0, 0),
]

# Small set for quick smoke tests.
SMOKE_TASKS: List[Tuple[str, int, int]] = [
    ("gsm8k", 5, 4),
    ("arc_easy", 0, 4),
]

# Agent-wrapped runs make sense for generation/reasoning tasks where
# code execution helps; avoid loglikelihood-style tasks.
AGENT_TASKS: List[Tuple[str, int, int]] = [
    ("gsm8k", 5, 0),
    ("triviaqa", 5, 0),
    ("drop", 3, 0),
]

AGENT_SMOKE_TASKS: List[Tuple[str, int, int]] = [
    ("gsm8k", 0, 4),
]

DEFAULT_MAX_TURNS = 4


def create_jsonl(
    name: str,
    tasks: List[Tuple[str, int, int]],
    template: str,
    include_max_turns: bool = False,
) -> None:
    os.makedirs(TASKS_DIR, exist_ok=True)
    out_path = os.path.join(TASKS_DIR, name + ".jsonl")
    with open(out_path, "wt") as fh:
        for task_name, fewshot, limit in tasks:
            substitutions = {
                "task_name.txt": {"__TASK_NAME__": task_name},
                "fewshot.txt": {"__FEWSHOT__": str(fewshot)},
                "limit.txt": {"__LIMIT__": str(limit)},
            }
            if include_max_turns:
                substitutions["max_turns.txt"] = {"__MAX_TURNS__": str(DEFAULT_MAX_TURNS)}
            record = {
                "id": task_name.replace("/", "_"),
                "template": template,
                "substitutions": substitutions,
            }
            fh.write(json.dumps(record).strip() + "\n")
    print(f"Wrote {out_path} ({len(tasks)} tasks)")


def main() -> None:
    # Direct (no agent) runs
    create_jsonl("lm_eval_default", DEFAULT_TASKS, TEMPLATE_DIRECT)
    create_jsonl("lm_eval_smoke", SMOKE_TASKS, TEMPLATE_DIRECT)
    # Agent-wrapped (coder + executor, multi-turn) runs
    create_jsonl("lm_eval_agent_default", AGENT_TASKS, TEMPLATE_AGENT, include_max_turns=True)
    create_jsonl("lm_eval_agent_smoke", AGENT_SMOKE_TASKS, TEMPLATE_AGENT, include_max_turns=True)


if __name__ == "__main__" and __package__ is None:
    main()
