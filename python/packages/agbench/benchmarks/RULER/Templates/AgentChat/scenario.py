"""
AgentChat template for RULER benchmark.
Single-turn: model answers a retrieval/reasoning question over long synthetic context.
Evaluation: exact substring match (case-insensitive) and F1 for multi-value tasks.
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

TASK_TYPE = "__TASK_TYPE__"

MULTI_VALUE_TASKS = {"niah_multivalue", "common_words"}


def normalize(s: str) -> str:
    """Lowercase, strip punctuation and whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s,]", "", s)
    return " ".join(s.split())


def evaluate(prediction: str, expected: str, task_type: str) -> dict:
    pred_norm = normalize(prediction)
    exp_norm = normalize(expected)

    if task_type in MULTI_VALUE_TASKS:
        # Parse comma-separated values
        pred_items = set(x.strip() for x in pred_norm.split(",") if x.strip())
        exp_items = set(x.strip() for x in exp_norm.split(",") if x.strip())
        if not exp_items:
            return {"passed": False, "score": 0.0, "metric": "set_f1"}
        correct = pred_items & exp_items
        precision = len(correct) / len(pred_items) if pred_items else 0.0
        recall = len(correct) / len(exp_items)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {
            "passed": recall == 1.0 and precision >= 0.5,
            "score": f1,
            "metric": "set_f1",
            "predicted_items": sorted(pred_items),
            "expected_items": sorted(exp_items),
        }
    else:
        # Single value: check if expected answer appears in prediction
        exact = pred_norm == exp_norm
        contained = exp_norm in pred_norm
        # Also check token-level F1
        pred_tokens = pred_norm.split()
        exp_tokens = exp_norm.split()
        common = Counter(pred_tokens) & Counter(exp_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            p = num_same / len(pred_tokens) if pred_tokens else 0.0
            r = num_same / len(exp_tokens) if exp_tokens else 0.0
            f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        return {
            "passed": exact or contained,
            "score": 1.0 if (exact or contained) else f1,
            "metric": "exact_or_contained",
        }


async def main() -> None:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(config["model_config"])

    model_context: ChatCompletionContext
    if model_client.model_info["family"] == ModelFamily.R1:
        model_context = ReasoningModelContext()
    else:
        model_context = UnboundedChatCompletionContext()

    start_time = time.time()

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        model_context=model_context,
        system_message=(
            "You are a helpful assistant. Answer the question precisely and concisely. "
            "Give only the answer itself with no explanation or extra words."
        ),
    )

    agent_team = RoundRobinGroupChat([agent], max_turns=6)

    with open("prompt.txt", "rt") as fh:
        prompt = fh.read()

    with open("expected_answer.txt", "rt") as fh:
        expected = fh.read().strip()

    stream = agent_team.run_stream(task=prompt)
    task_result = await Console(stream)

    end_time = time.time()
    print(f"AgentChat execution time: {end_time - start_time:.2f} seconds")

    # Extract prediction
    prediction = ""
    for msg in reversed(task_result.messages):
        if msg.source == "assistant":
            prediction = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    result = evaluate(prediction, expected, TASK_TYPE)
    print(f"\n[evaluation] task_type={TASK_TYPE} metric={result['metric']} score={result['score']:.4f}")
    print(f"[evaluation] prediction={prediction[:200]}")
    print(f"[evaluation] expected={expected[:200]}")

    if result["passed"]:
        print("ALL TESTS PASSED !#!#")
    else:
        print("SOME TESTS FAILED !#!#")

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
