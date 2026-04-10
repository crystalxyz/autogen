#
# Run this file to download the DS-1000 dataset, and create corresponding testbed scenarios.
#

import json
import os
import re

from datasets import load_dataset

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")


def download_ds1000():
    """Download the DS-1000 dataset from HuggingFace and return a list of parsed task dicts."""
    ds = load_dataset("xlangai/DS-1000", split="test")
    results = []
    for i, row in enumerate(ds):
        metadata = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
        lib = metadata.get("library", "unknown")
        task_id = f"DS1000_{lib}_{i}"
        results.append(
            {
                "task_id": task_id,
                "prompt": row["prompt"],
                "reference_code": row["reference_code"],
                "code_context": row.get("code_context", ""),
                "test": row.get("test", ""),
                "lib": lib,
                "metadata": metadata,
            }
        )
    return results


def create_jsonl(name, tasks, template):
    """Creates a JSONL scenario file with a given name, list of DS-1000 tasks, and template path."""

    # Create a task directory if it doesn't exist
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    # Create the jsonl file
    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['task_id']}")

            record = {
                "id": task["task_id"],
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": task["prompt"]},
                    "test.txt": {"__TEST__": task["test"]},
                    "code_context.txt": {"__CODE_CONTEXT__": task["code_context"]},
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def main():
    ds1000 = download_ds1000()
    print(f"Downloaded {len(ds1000)} DS-1000 tasks")

    # list all directories in the Templates directory
    # and populate a dictionary with the name and path
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    # Create the various combinations of [templates]
    for t in templates.items():
        create_jsonl(f"ds1000_{t[0]}", ds1000, t[1])


if __name__ == "__main__" and __package__ is None:
    main()
