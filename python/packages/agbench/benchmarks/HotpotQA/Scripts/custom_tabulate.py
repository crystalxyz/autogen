import os
import sys
import re
import string
from collections import Counter
from typing import Optional

from agbench.tabulate_cmd import default_tabulate


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.

    Based on the official HotpotQA evaluation script.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if normalized prediction matches normalized ground truth exactly."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def scorer(instance_dir: str):
    """Score a single HotpotQA instance.

    Args:
        instance_dir: Path to the instance directory containing expected_answer.txt
                      and console_log.txt

    Returns:
        True if exact match, False otherwise, None if files missing
    """
    # Read the expected answer
    expected_answer_file = os.path.join(instance_dir, "expected_answer.txt")
    if not os.path.isfile(expected_answer_file):
        return None

    with open(expected_answer_file, "rt") as fh:
        expected_answer = fh.read().strip()

    # Read the console log
    console_log_file = os.path.join(instance_dir, "console_log.txt")
    if not os.path.isfile(console_log_file):
        return None

    with open(console_log_file, "rt") as fh:
        console_log = fh.read()

    # Extract the final answer
    final_answer = None

    # Try to find FINAL ANSWER pattern (use last match — earlier matches may come
    # from serialized system-prompt instructions in the chat history)
    matches = re.findall(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", console_log, re.IGNORECASE | re.DOTALL)
    if matches:
        final_answer = matches[-1].strip()

    # Also try ANSWER: pattern
    if final_answer is None:
        matches = re.findall(r"ANSWER:\s*(.+?)(?:\n|$)", console_log, re.IGNORECASE | re.DOTALL)
        if matches:
            final_answer = matches[-1].strip()

    # Missing the final answer
    if final_answer is None:
        return None

    # Return exact match result
    return exact_match_score(final_answer, expected_answer)


def f1_scorer(instance_dir: str) -> Optional[float]:
    """Return F1 score (0.0–1.0) for a single instance, or None if missing."""
    expected_answer_file = os.path.join(instance_dir, "expected_answer.txt")
    console_log_file = os.path.join(instance_dir, "console_log.txt")

    if not os.path.isfile(expected_answer_file) or not os.path.isfile(console_log_file):
        return None

    with open(expected_answer_file, "rt") as fh:
        expected_answer = fh.read().strip()

    with open(console_log_file, "rt") as fh:
        console_log = fh.read()

    matches = re.findall(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", console_log, re.IGNORECASE | re.DOTALL)
    if not matches:
        matches = re.findall(r"ANSWER:\s*(.+?)(?:\n|$)", console_log, re.IGNORECASE | re.DOTALL)
    if not matches:
        return None

    return f1_score(matches[-1].strip(), expected_answer)


def main(args):
    default_tabulate(args, scorer=scorer)

    # Also compute and print average F1 across all instances
    import argparse as _ap

    parser = _ap.ArgumentParser(add_help=False)
    parser.add_argument("runlogs")
    parsed, _ = parser.parse_known_args(args[1:])
    runlogs = parsed.runlogs

    exclude = {"result.json", "result.json.tmp", "__pycache__"}
    f1_scores, em_scores = [], []
    for task_id in sorted(os.listdir(runlogs)):
        if task_id in exclude:
            continue
        task_path = os.path.join(runlogs, task_id)
        if not os.path.isdir(task_path):
            continue
        for rep in sorted(os.listdir(task_path)):
            rep_path = os.path.join(task_path, rep)
            if not rep.isdigit() or not os.path.isdir(rep_path):
                continue
            f1 = f1_scorer(rep_path)
            em = scorer(rep_path)
            if f1 is not None:
                f1_scores.append(f1)
            if em is not None:
                em_scores.append(1 if em else 0)

    if f1_scores or em_scores:
        n = max(len(f1_scores), len(em_scores))
        print(f"\nHotpotQA Score Summary ({n} instances):")
        if f1_scores:
            print(f"  Average F1:    {sum(f1_scores)/len(f1_scores):.4f}")
        if em_scores:
            print(f"  Exact Match:   {sum(em_scores)/len(em_scores):.4f}")


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
