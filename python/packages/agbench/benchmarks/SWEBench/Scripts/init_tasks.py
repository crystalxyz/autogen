#
# Run this file to download the SWE-bench dataset and create corresponding testbed scenario files.
# Downloads SWE-bench_Lite by default (300 tasks), or the full SWE-bench dataset.
#

import json
import os
import re
import sys

from datasets import load_dataset

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")
DOWNLOADS_DIR = os.path.join(SCENARIO_DIR, "Downloads")


def download_swebench(lite=True):
    """Download SWE-bench dataset from Hugging Face."""
    dataset_id = "princeton-nlp/SWE-bench_Lite" if lite else "princeton-nlp/SWE-bench"
    print(f"Downloading {dataset_id} ...")
    dataset = load_dataset(dataset_id)
    return dataset


def create_jsonl(name, tasks, template):
    """Creates a JSONL scenario file with a given name, list of SWE-bench tasks, and template path."""

    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['instance_id']}")

            # Sanitize instance_id for use as a directory name
            task_id = re.sub(r"[^\w\-]", "_", task["instance_id"])

            record = {
                "id": task_id,
                "template": template,
                "substitutions": {
                    "scenario.py": {
                        "__INSTANCE_ID__": task["instance_id"],
                        "__REPO__": task["repo"],
                        "__BASE_COMMIT__": task["base_commit"],
                        "__FAIL_TO_PASS__": task["FAIL_TO_PASS"],
                        "__PASS_TO_PASS__": task["PASS_TO_PASS"],
                        "__VERSION__": task.get("version", ""),
                    },
                    "prompt.txt": {
                        "__PROBLEM_STATEMENT__": task["problem_statement"],
                    },
                    "expected_answer.txt": {
                        "__EXPECTED_ANSWER__": task.get("patch", ""),
                    },
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


###############################################################################
def main():
    # Parse arguments
    lite = True
    if "--full" in sys.argv:
        lite = False

    dataset = download_swebench(lite=lite)

    suffix = "lite" if lite else "full"

    # List all template directories
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    # Create JSONL for each split and template combination
    for split in dataset.keys():
        tasks = list(dataset[split])
        for t in templates.items():
            create_jsonl(f"swebench_{suffix}_{split}__{t[0]}", tasks, t[1])


if __name__ == "__main__" and __package__ is None:
    main()
