#
# Run this file to download the LongBench dataset and create corresponding testbed scenarios.
#

import json
import os
import re
import zipfile

from huggingface_hub import hf_hub_download

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_NAME = os.path.basename(SCRIPT_PATH)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)

SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")

# English tasks (skip Chinese and *_e variants)
ENGLISH_TASKS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]


def download_longbench():
    """Download LongBench data.zip from HuggingFace and return path."""
    print("Downloading LongBench data.zip from HuggingFace...")
    path = hf_hub_download(repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset")
    return path


def load_task(zip_path: str, task_name: str):
    """Load a specific task's JSONL from the zip file."""
    filename = f"data/{task_name}.jsonl"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(filename) as f:
            return [json.loads(line) for line in f]


def create_jsonl(name: str, tasks: list, template: str, task_name: str):
    """Creates a JSONL scenario file."""
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for i, task in enumerate(tasks):
            task_id = f"LongBench_{task_name}_{i}"
            print(f"Converting: [{name}] {task_id}")

            # Build the prompt with context
            context = task.get("context", "")
            input_text = task.get("input", "")
            all_classes = task.get("all_classes", None)

            if context:
                full_prompt = f"Context:\n{context}\n\nQuestion: {input_text}"
            else:
                full_prompt = input_text

            # For classification tasks, include the valid classes
            if all_classes:
                if isinstance(all_classes, str):
                    all_classes = json.loads(all_classes)
                if all_classes:
                    classes_str = ", ".join(str(c) for c in all_classes)
                    full_prompt += f"\n\nValid categories: {classes_str}"

            # answers is a list of acceptable answers
            answers = task.get("answers", [])
            if isinstance(answers, str):
                answers = json.loads(answers)
            answers_json = json.dumps(answers)

            record = {
                "id": task_id,
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": full_prompt},
                    "expected_answer.txt": {"__EXPECTED_ANSWER__": answers_json},
                    "scenario.py": {"__TASK_TYPE__": task_name},
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


def main():
    # List all directories in the Templates directory
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    if not templates:
        print("No templates found in Templates directory.")
        return

    # Download the zip once
    zip_path = download_longbench()

    # Process English tasks
    for task_name in ENGLISH_TASKS:
        try:
            tasks = load_task(zip_path, task_name)
            print(f"  Loaded {len(tasks)} examples for {task_name}")
        except Exception as e:
            print(f"  Error loading {task_name}: {e}")
            continue

        for template_name, template_path in templates.items():
            create_jsonl(
                f"longbench_{task_name}_{template_name}",
                tasks,
                template_path,
                task_name,
            )

    print(f"\nDone! Task files created in: {TASKS_DIR}")


if __name__ == "__main__" and __package__ is None:
    main()
