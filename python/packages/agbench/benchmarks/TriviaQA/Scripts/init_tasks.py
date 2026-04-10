#
# Run this file to download the TriviaQA dataset, and create corresponding testbed scenarios.
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


def download_triviaqa(config_name: str = "rc"):
    """Download the TriviaQA dataset from Hugging Face.

    Args:
        config_name: 'rc' (reading comprehension with evidence) or
                     'unfiltered' (noisy evidence). Default 'rc'.

    Returns:
        Dictionary with 'train' and 'validation' splits
    """
    print(f"Downloading TriviaQA ({config_name}) from Hugging Face...")
    dataset = load_dataset("trivia_qa", config_name, trust_remote_code=True)
    return dataset


def _truncate_text(text: str, char_limit: int) -> str:
    """Truncate text to char_limit, cutting at the last sentence boundary."""
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    cut = text[:char_limit]
    last_period = cut.rfind(".")
    if last_period > char_limit // 4:
        return cut[: last_period + 1]
    return cut + "..."


def format_evidence(
    task: dict,
    max_paragraphs: int = 10,
    char_limit: int = 2000,
    include_search_results: bool = False,
) -> str:
    """Format the evidence documents for the prompt.

    TriviaQA provides evidence in two forms:
      - entity_pages: Wikipedia pages related to the answer entity
      - search_results: web search snippets

    Args:
        task: A single TriviaQA example dict
        max_paragraphs: Maximum number of evidence paragraphs to include
        char_limit: Maximum characters per paragraph (0 = no limit)
        include_search_results: If True, append search results after entity
            pages (not just as a fallback). Useful for stress-testing context.

    Returns:
        Formatted evidence string, or empty string if no evidence
    """
    formatted_parts = []
    idx = 1

    entity_pages = task.get("entity_pages", {})
    titles = entity_pages.get("title", [])
    wiki_contexts = entity_pages.get("wiki_context", [])

    for title, wiki_context in zip(titles, wiki_contexts):
        if idx > max_paragraphs:
            break
        if not wiki_context or not wiki_context.strip():
            continue
        text = wiki_context.strip()
        if char_limit > 0:
            text = _truncate_text(text, char_limit)
        formatted_parts.append(f"[{idx}] {title}\n{text}")
        idx += 1

    # Include search results: always if flag set, otherwise only as fallback
    add_search = include_search_results or (not formatted_parts)
    if add_search:
        search_results = task.get("search_results", {})
        sr_titles = search_results.get("title", [])
        sr_snippets = search_results.get("search_context", [])

        for title, snippet in zip(sr_titles, sr_snippets):
            if idx > max_paragraphs:
                break
            if not snippet or not snippet.strip():
                continue
            text = snippet.strip()
            if char_limit > 0:
                text = _truncate_text(text, char_limit)
            formatted_parts.append(f"[{idx}] {title}\n{text}")
            idx += 1

    return "\n\n".join(formatted_parts)


def create_jsonl(
    name: str,
    tasks: list,
    template: str,
    max_paragraphs: int = 10,
    char_limit: int = 2000,
    include_search_results: bool = False,
):
    """Creates a JSONL scenario file with a given name and template path.

    Args:
        name: Name for the output JSONL file
        tasks: List of TriviaQA task dictionaries
        template: Path to the template directory
        max_paragraphs: Maximum evidence paragraphs per task
        char_limit: Max chars per paragraph (0 = no limit)
        include_search_results: Include web search results alongside Wikipedia
    """
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            task_id = task["question_id"]
            print(f"Converting: [{name}] {task_id}")

            question = task["question"]
            evidence_str = format_evidence(
                task,
                max_paragraphs=max_paragraphs,
                char_limit=char_limit,
                include_search_results=include_search_results,
            )

            if evidence_str:
                full_prompt = f"Evidence:\n{evidence_str}\n\nQuestion: {question}"
            else:
                full_prompt = f"Question: {question}"

            # Store answer with aliases for multi-alias evaluation
            answer_data = {
                "value": task["answer"]["value"],
                "aliases": task["answer"].get("aliases", []),
            }

            record = {
                "id": task_id,
                "template": template,
                "substitutions": {
                    "prompt.txt": {"__PROMPT__": full_prompt},
                    "expected_answer.txt": {
                        "__EXPECTED_ANSWER__": json.dumps(answer_data)
                    },
                },
            }

            fh.write(json.dumps(record).strip() + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Download TriviaQA and create task JSONL files.")
    parser.add_argument(
        "--max-paragraphs", type=int, default=10,
        help="Max evidence paragraphs per task (default: 10)",
    )
    parser.add_argument(
        "--char-limit", type=int, default=2000,
        help="Max chars per paragraph, 0 = no limit (default: 2000)",
    )
    parser.add_argument(
        "--include-search-results", action="store_true",
        help="Include web search results alongside Wikipedia pages",
    )
    parser.add_argument(
        "--suffix", type=str, default="",
        help="Extra suffix appended to output JSONL filenames",
    )
    args = parser.parse_args()

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

    suffix = f"_{args.suffix}" if args.suffix else ""

    # Download TriviaQA (rc = reading comprehension with evidence documents)
    try:
        dataset = download_triviaqa("rc")
    except Exception as e:
        print(f"Error downloading TriviaQA: {e}")
        return

    evidence_kwargs = dict(
        max_paragraphs=args.max_paragraphs,
        char_limit=args.char_limit,
        include_search_results=args.include_search_results,
    )

    # Create JSONL files for each template and split
    for template_name, template_path in templates.items():
        # Validation split
        if "validation" in dataset:
            validation_tasks = list(dataset["validation"])
            create_jsonl(
                f"triviaqa_validation_rc__{template_name}{suffix}",
                validation_tasks,
                template_path,
                **evidence_kwargs,
            )
            # Sanity subset (100 tasks) for quick testing
            create_jsonl(
                f"triviaqa_validation_rc__{template_name}{suffix}_sanity100",
                validation_tasks[:100],
                template_path,
                **evidence_kwargs,
            )

        # Train split (optional, usually not run in full)
        if "train" in dataset:
            train_tasks = list(dataset["train"])
            create_jsonl(
                f"triviaqa_train_rc__{template_name}{suffix}",
                train_tasks,
                template_path,
                **evidence_kwargs,
            )

    print("\nDone! Task files created in:", TASKS_DIR)


if __name__ == "__main__" and __package__ is None:
    main()
