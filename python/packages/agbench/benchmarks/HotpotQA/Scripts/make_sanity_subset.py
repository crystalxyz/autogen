"""Generate a deterministic, seeded sanity subset of HotpotQA validation tasks
for the SelectorGroupChat benchmark template.

Selection criteria:
    - Source:   hotpotqa_validation_fullwiki__ReActFullWiki.jsonl (7,405 tasks)
                which is the full HotpotQA validation split in fullwiki (open-domain) mode.
    - Sample:   uniform random without replacement, seeded with SEED below.
    - Size:     SUBSET_SIZE tasks.
    - Template: replaced from ReActFullWiki to SelectorGroupChat (same id/question/answer).

To regenerate identically, rerun this script with the same SEED and SUBSET_SIZE.

Output:
    benchmarks/HotpotQA/Tasks/hotpotqa_validation_fullwiki__SelectorGroupChat_sanity{N}_seed{S}.jsonl
"""

import json
import os
import random
import sys

SEED = 42
SUBSET_SIZE = 100

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")

SOURCE = os.path.join(TASKS_DIR, "hotpotqa_validation_fullwiki__ReActFullWiki.jsonl")
TARGET_TEMPLATE = os.path.join(TEMPLATES_DIR, "SelectorGroupChat")
OUTPUT = os.path.join(
    TASKS_DIR,
    f"hotpotqa_validation_fullwiki__SelectorGroupChat_sanity{SUBSET_SIZE}_seed{SEED}.jsonl",
)


def main() -> int:
    if not os.path.isfile(SOURCE):
        print(f"Source file not found: {SOURCE}", file=sys.stderr)
        print("Run benchmarks/HotpotQA/Scripts/init_tasks.py first to generate it.", file=sys.stderr)
        return 1

    if not os.path.isdir(TARGET_TEMPLATE):
        print(f"Template directory not found: {TARGET_TEMPLATE}", file=sys.stderr)
        return 1

    with open(SOURCE, "rt") as fh:
        tasks = [json.loads(line) for line in fh]

    print(f"Source: {len(tasks)} tasks in {os.path.basename(SOURCE)}")

    rng = random.Random(SEED)
    # sorted() makes input-order-independent; random.Random(SEED).sample is
    # deterministic given identical input.
    tasks_sorted = sorted(tasks, key=lambda t: t["id"])
    sampled = rng.sample(tasks_sorted, SUBSET_SIZE)

    os.makedirs(TASKS_DIR, exist_ok=True)
    with open(OUTPUT, "wt") as fh:
        for t in sampled:
            record = dict(t)
            record["template"] = TARGET_TEMPLATE
            fh.write(json.dumps(record) + "\n")

    print(f"Wrote {SUBSET_SIZE} tasks to: {OUTPUT}")
    print(f"Seed: {SEED}   (change SEED to produce a different subset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
