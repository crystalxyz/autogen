#
# Run this file to download the HotpotQA dataset, and create corresponding testbed scenarios.
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
DOWNLOADS_DIR = os.path.join(SCENARIO_DIR, "Downloads")


def download_hotpotqa(config_name: str = "distractor"):
    """Download the HotpotQA dataset from Hugging Face.

    Args:
        config_name: Either 'distractor' or 'fullwiki'

    Returns:
        Dictionary with 'train' and 'validation' splits
    """
    print(f"Downloading HotpotQA ({config_name}) from Hugging Face...")
    dataset = load_dataset("hotpotqa/hotpot_qa", config_name, trust_remote_code=True)
    return dataset


def format_context(context: dict) -> str:
    """Format the context paragraphs for the prompt.

    Args:
        context: Dictionary with 'title' and 'sentences' lists

    Returns:
        Formatted context string
    """
    formatted_parts = []
    titles = context.get("title", [])
    sentences_list = context.get("sentences", [])

    for i, (title, sentences) in enumerate(zip(titles, sentences_list)):
        paragraph = "".join(sentences)
        formatted_parts.append(f"[{i+1}] {title}\n{paragraph}")

    return "\n\n".join(formatted_parts)


def create_jsonl(name: str, tasks: list, template: str, include_context: bool = True):
    """Creates a JSONL scenario file with a given name and template path.

    Args:
        name: Name for the output JSONL file
        tasks: List of HotpotQA task dictionaries
        template: Path to the template directory
        include_context: Whether to include context paragraphs in the prompt
    """
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            task_id = task["id"]
            print(f"Converting: [{name}] {task_id}")

            # Build the prompt
            question = task["question"]
            if include_context and "context" in task:
                context_str = format_context(task["context"])
                full_prompt = f"Context:\n{context_str}\n\nQuestion: {question}"
            else:
                full_prompt = f"Question: {question}"

            record = {
                "id": task_id,
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": full_prompt},
                    "expected_answer.txt": {"__EXPECTED_ANSWER__": task["answer"]},
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


def main():
    # Ensure downloads directory exists
    if not os.path.isdir(DOWNLOADS_DIR):
        os.mkdir(DOWNLOADS_DIR)

    # List all directories in the Templates directory
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    if not templates:
        print("No templates found in Templates directory. Please create at least one template.")
        return

    # Process both configurations
    for config_name in ["distractor", "fullwiki"]:
        try:
            dataset = download_hotpotqa(config_name)
        except Exception as e:
            print(f"Error downloading {config_name}: {e}")
            continue

        # Include context for distractor, not for fullwiki (open-domain)
        include_context = (config_name == "distractor")

        # Create JSONL files for each template and split
        for template_name, template_path in templates.items():
            # Validation split
            if "validation" in dataset:
                validation_tasks = list(dataset["validation"])
                create_jsonl(
                    f"hotpotqa_validation_{config_name}__{template_name}",
                    validation_tasks,
                    template_path,
                    include_context=include_context,
                )

            # Train split (optional, usually not run in full)
            if "train" in dataset:
                train_tasks = list(dataset["train"])
                create_jsonl(
                    f"hotpotqa_train_{config_name}__{template_name}",
                    train_tasks,
                    template_path,
                    include_context=include_context,
                )

    print("\nDone! Task files created in:", TASKS_DIR)


if __name__ == "__main__" and __package__ is None:
    main()
