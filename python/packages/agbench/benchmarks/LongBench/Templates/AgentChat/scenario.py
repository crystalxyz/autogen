"""
AgentChat template for LongBench benchmark.
Single-turn QA: model answers a question given long context.
Evaluation: F1 score and exact match (QA tasks), ROUGE-L (summarization),
            or accuracy (classification/synthetic tasks).
"""

import asyncio
import json
import re
import string
import time
from collections import Counter

import yaml
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.model_context import ChatCompletionContext, UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient, ModelFamily
from reasoning_model_context import ReasoningModelContext

# Task type is substituted per-problem by init_tasks.py
TASK_TYPE = "__TASK_TYPE__"

# Task categories for metric selection
SUMMARIZATION_TASKS = {"gov_report", "qmsum", "multi_news", "vcsum"}
CLASSIFICATION_TASKS = {"trec", "lsht"}
SYNTHETIC_TASKS = {"passage_count", "passage_retrieval_en", "passage_retrieval_zh"}
CODE_TASKS = {"lcc", "repobench-p"}
# Everything else is QA (F1/EM)


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
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

    return (2 * precision * recall) / (precision + recall)


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 score using longest common subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via dynamic programming
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    precision = lcs_len / m if m else 0.0
    recall = lcs_len / n if n else 0.0

    if precision + recall == 0:
        return 0.0

    return (2 * precision * recall) / (precision + recall)


def evaluate(prediction: str, answers: list, task_type: str) -> dict:
    """Evaluate prediction against ground truth answers.

    Returns dict with: passed, score, metric, details
    """
    if not answers:
        return {"passed": False, "score": 0.0, "metric": "none", "details": "No answers provided"}

    if task_type in SUMMARIZATION_TASKS:
        # ROUGE-L: take best score across answers
        best_score = max(rouge_l_score(prediction, ans) for ans in answers)
        return {
            "passed": best_score >= 0.15,
            "score": best_score,
            "metric": "rouge_l",
        }
    elif task_type in CLASSIFICATION_TASKS or task_type in SYNTHETIC_TASKS:
        # Accuracy: exact match after normalization
        pred_norm = normalize_answer(prediction)
        for ans in answers:
            if normalize_answer(ans) == pred_norm:
                return {"passed": True, "score": 1.0, "metric": "accuracy"}
        return {"passed": False, "score": 0.0, "metric": "accuracy"}
    elif task_type in CODE_TASKS:
        # Edit similarity (simplified: use F1 as proxy)
        best_score = max(f1_score(prediction, ans) for ans in answers)
        return {
            "passed": best_score >= 0.3,
            "score": best_score,
            "metric": "code_f1",
        }
    else:
        # QA tasks: F1 score, take best across answers
        best_f1 = max(f1_score(prediction, ans) for ans in answers)
        best_em = any(normalize_answer(prediction) == normalize_answer(ans) for ans in answers)
        passed = best_em or best_f1 >= 0.5
        return {
            "passed": passed,
            "score": best_f1,
            "metric": "f1",
            "exact_match": best_em,
        }


async def main() -> None:
    # Load model configuration and create the model client.
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(config["model_config"])

    # Model context
    model_context: ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    start_time = time.time()

    # Single assistant agent for answering
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_context=model_context,
        system_message=(
            "You are a helpful assistant. Answer the question based on the provided context. "
            "Be concise and direct. Provide only the answer without explanation unless asked."
        ),
    )

    # Single turn: just one response from the model
    agent_team = RoundRobinGroupChat([agent], max_turns=1)

    # Load prompt and expected answer
    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    with open("expected_answer.txt", "rt") as fh:
        answers_raw = fh.read().strip()

    # Parse answers (stored as JSON list)
    try:
        answers = json.loads(answers_raw)
        if isinstance(answers, str):
            answers = [answers]
    except json.JSONDecodeError:
        answers = [answers_raw]

    # Run the team
    stream = agent_team.run_stream(task=prompt)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Extract the model's answer (last assistant message)
    prediction = ""
    for msg in reversed(task_result.messages):
        if msg.source == "assistant":
            prediction = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    # Evaluate
    result = evaluate(prediction, answers, TASK_TYPE)
    print(f"\n[evaluation] task_type={TASK_TYPE} metric={result['metric']} score={result['score']:.4f}")
    print(f"[evaluation] prediction={prediction[:200]}")
    print(f"[evaluation] expected={answers[0][:200] if answers else 'N/A'}")

    if result["passed"]:
        print("ALL TESTS PASSED !#!#")
    else:
        print("SOME TESTS FAILED !#!#")

    # Print token usage
    turn = 0
    for msg in task_result.messages:
        if msg.models_usage is not None:
            turn += 1
            print(
                f"[token_usage] turn={turn} source={msg.source}"
                f" prompt_tokens={msg.models_usage.prompt_tokens}"
                f" completion_tokens={msg.models_usage.completion_tokens}"
                f" reasoning_tokens={msg.models_usage.reasoning_tokens}"
            )


asyncio.run(main())
