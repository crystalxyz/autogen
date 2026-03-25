import os
import sys
import re
import string
from collections import Counter

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


def main(args):
    default_tabulate(args, scorer=scorer)


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
