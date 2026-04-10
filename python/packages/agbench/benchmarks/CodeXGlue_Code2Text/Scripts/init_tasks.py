"""
Download the CodeXGlue code-to-text (Python) dataset from HuggingFace
and create agbench task JSONL files.

Usage:
    python Scripts/init_tasks.py [--max-tasks N] [--lang LANG] [--split SPLIT]
"""

import argparse
import json
import os
import re

import ast
import textwrap

from datasets import load_dataset

SCRIPT_PATH = os.path.realpath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
SCENARIO_DIR = os.path.realpath(os.path.join(SCRIPT_DIR, os.path.pardir))
TEMPLATES_DIR = os.path.join(SCENARIO_DIR, "Templates")
TASKS_DIR = os.path.join(SCENARIO_DIR, "Tasks")


def strip_docstring(code: str, docstring: str) -> str:
    """Remove the docstring from a Python function's code.

    Tries AST-based removal first, falls back to string replacement.
    """
    # Try AST-based approach
    try:
        tree = ast.parse(textwrap.dedent(code))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, (ast.Constant, ast.Str))):
                    # Get line range of the docstring
                    ds_node = body[0]
                    lines = code.splitlines(keepends=True)
                    # Remove docstring lines (1-indexed to 0-indexed)
                    start = ds_node.lineno - 1
                    end = ds_node.end_lineno  # type: ignore[attr-defined]
                    cleaned = lines[:start] + lines[end:]
                    result = "".join(cleaned)
                    # Remove blank lines right after function def
                    result = re.sub(r"(def\s+\w+\(.*?\)\s*:)\s*\n(\s*\n)+", r"\1\n", result)
                    return result
    except Exception:
        pass

    # Fallback: try to remove triple-quoted docstring via regex
    for quote in ['"""', "'''"]:
        pattern = re.escape(quote) + r".*?" + re.escape(quote)
        cleaned = re.sub(pattern, "", code, count=1, flags=re.DOTALL)
        if cleaned != code:
            return cleaned.strip()

    return code


def create_jsonl(name, tasks, template):
    """Creates a JSONL scenario file with a given name, list of tasks, and template path."""
    if not os.path.isdir(TASKS_DIR):
        os.mkdir(TASKS_DIR)

    with open(os.path.join(TASKS_DIR, name + ".jsonl"), "wt") as fh:
        for task in tasks:
            print(f"Converting: [{name}] {task['id']}")

            record = {
                "id": task["id"],
                "template": template,
                "substitutions": {
                    "scenario.py": {
                        "__CODE__": task["code"],
                        "__REFERENCE__": task["docstring"],
                    },
                },
            }
            fh.write(json.dumps(record).strip() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=200, help="Maximum number of tasks to generate")
    parser.add_argument("--lang", type=str, default="python", help="Language config (python, java, go, php, javascript, ruby)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, validation, test)")
    args = parser.parse_args()

    print(f"Loading CodeXGlue code-to-text ({args.lang}) from HuggingFace...")
    ds = load_dataset("google/code_x_glue_ct_code_to_text", args.lang, split=args.split)

    # Build task list (cap at max_tasks)
    tasks = []
    for i, row in enumerate(ds):
        if i >= args.max_tasks:
            break

        # Sanitize func_name for use as task ID
        func_name = re.sub(r"[^a-zA-Z0-9_]", "_", row["func_name"])
        task_id = f"CodeXGlue_{args.lang}_{i}_{func_name}"

        # Strip docstring from code so the model can't just copy it
        code = strip_docstring(row["code"], row["docstring"])

        tasks.append({
            "id": task_id,
            "code": code,
            "docstring": row["docstring"],
        })

    print(f"Loaded {len(tasks)} tasks from {args.split} split")

    # Discover templates and create JSONL for each
    templates = {}
    for entry in os.scandir(TEMPLATES_DIR):
        if entry.is_dir():
            templates[re.sub(r"\s", "", entry.name)] = entry.path

    for t_name, t_path in templates.items():
        create_jsonl(f"code2text_{args.lang}_{t_name}", tasks, t_path)


if __name__ == "__main__" and __package__ is None:
    main()
