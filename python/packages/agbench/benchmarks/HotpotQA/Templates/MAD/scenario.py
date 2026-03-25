"""
Multi-Agent Debate (MAD) template for HotpotQA benchmark.

Architecture:
    Debaters -> Judge (select OR synthesize) -> Evaluator (F1/EM) -> Result

Judge Modes:
    - DISCRIMINATIVE: Judge selects the best answer from candidates
    - EXTRACTIVE: Judge synthesizes a new answer combining best reasoning
"""

import asyncio
import re
import string
import time
from collections import Counter
from typing import Any

import yaml
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.models import ChatCompletionClient

from agbench.agents.debating_agent import (
    DebateTask,
    DiscriminativeJudge,
    EvaluationResult,
    ExtractiveJudge,
    JudgeMode,
    ReasoningDomainConfig,
    SolutionEvaluator,
    TopologyType,
    TopologyConfig,
    Debater,
    DebateAggregator,
    ResultCollector,
)


class HotpotQAEvaluator(SolutionEvaluator):
    """Evaluator using HotpotQA F1 and Exact Match metrics."""

    def __init__(self, expected_answer: str | None = None) -> None:
        self._expected_answer = expected_answer

    def _normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _f1_score(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score."""
        prediction_tokens = self._normalize_answer(prediction).split()
        ground_truth_tokens = self._normalize_answer(ground_truth).split()

        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = num_same / len(prediction_tokens) if prediction_tokens else 0.0
        recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0.0

        if precision + recall == 0:
            return 0.0

        return (2 * precision * recall) / (precision + recall)

    def _exact_match(self, prediction: str, ground_truth: str) -> bool:
        """Check if normalized prediction matches normalized ground truth."""
        return self._normalize_answer(prediction) == self._normalize_answer(ground_truth)

    async def evaluate(self, solution: str, task: DebateTask) -> EvaluationResult:
        expected = self._expected_answer or task.context.get("expected_answer")
        if expected is None:
            return EvaluationResult(passed=False, error="No expected answer provided")

        f1 = self._f1_score(solution, expected)
        em = self._exact_match(solution, expected)

        # Consider passed if F1 >= 0.5 or exact match
        passed = em or f1 >= 0.5

        return EvaluationResult(
            passed=passed,
            score=f1,
            details={
                "prediction": solution[:200],
                "expected": expected[:200],
                "f1": f1,
                "exact_match": em,
            },
        )


async def run_debate_with_custom_prompts(
    task: str,
    model_client: ChatCompletionClient,
    domain_config: ReasoningDomainConfig,
    judge: DiscriminativeJudge | ExtractiveJudge,
    evaluator: SolutionEvaluator,
    system_prompts: dict[str, str],
    num_debaters: int = 4,
    max_rounds: int = 2,
    topology_type: TopologyType = TopologyType.CIRCULAR,
    context: dict[str, Any] | None = None,
):
    """Run debate with custom system prompts per debater."""
    context = context or {}

    # Setup topology
    topology = TopologyConfig(topology_type=topology_type, num_debaters=num_debaters)
    debater_ids = topology.get_debater_ids()
    connections = topology.get_connections()

    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    # Register debaters with custom system prompts
    for debater_id in debater_ids:
        num_neighbors = topology.get_num_neighbors(debater_id)
        system_prompt = system_prompts.get(debater_id)  # None uses domain default

        await Debater.register(
            runtime,
            debater_id,
            lambda did=debater_id, nn=num_neighbors, sp=system_prompt: Debater(
                model_client=model_client,
                domain_config=domain_config,
                topic_type=did,
                num_neighbors=nn,
                max_rounds=max_rounds,
                debater_id=did,
                system_prompt=sp,
            ),
        )

    # Register aggregator
    await DebateAggregator.register(
        runtime,
        "Aggregator",
        lambda: DebateAggregator(
            domain_config=domain_config,
            judge=judge,
            evaluator=evaluator,
            num_debaters=len(debater_ids),
        ),
    )

    # Register result collector
    await ResultCollector.register(runtime, "ResultCollector", ResultCollector)

    # Setup topology subscriptions
    for debater_id, neighbors in connections.items():
        for neighbor_id in neighbors:
            await runtime.add_subscription(TypeSubscription(debater_id, neighbor_id))

    # Run debate
    runtime.start()
    await runtime.publish_message(
        DebateTask(task_id="task_001", prompt=task, context=context),
        topic_id=DefaultTopicId(),
    )
    await runtime.stop_when_idle()

    # Get result
    collector = await runtime.try_get_underlying_agent_instance(
        await runtime.get("ResultCollector", "default"),
        ResultCollector,
    )

    if collector.result is None:
        raise RuntimeError("No result collected")

    return collector.result


async def main() -> None:
    start_time = time.time()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("prompt.txt", "r") as f:
        prompt = f.read().strip()

    with open("expected_answer.txt", "r") as f:
        expected_answer = f.read().strip()

    model_client = ChatCompletionClient.load_component(config["model_config"])

    # Configuration
    num_debaters = 4
    max_rounds = 2
    judge_mode = JudgeMode.DISCRIMINATIVE

    print(f"Multi-Agent Debate: {num_debaters} debaters, {max_rounds} rounds, {judge_mode.value} judge")
    print("=" * 60)

    # Setup domain and evaluator
    domain = ReasoningDomainConfig()
    judge = (
        DiscriminativeJudge(model_client, domain)
        if judge_mode == JudgeMode.DISCRIMINATIVE
        else ExtractiveJudge(model_client, domain)
    )
    evaluator = HotpotQAEvaluator(expected_answer=expected_answer)

    # Define custom system prompts for each debater
    # You can customize these to give each debater a different perspective
    system_prompts = {
        "DebaterA": """You are a logical reasoning expert participating in collaborative problem solving.
Your task is to analyze problems carefully and provide well-reasoned answers.

When solving problems:
1. Carefully read and understand the question
2. Break down complex problems into parts
3. Consider multiple perspectives

IMPORTANT: Format your final answer on its own line starting with: ANSWER:

When reviewing other solutions: You are affirmative side. Please express your viewpoints.""",
        "DebaterB": """You are a fact-checking specialist participating in collaborative problem solving.
Your task is to verify claims and ensure accuracy in answers.

When solving problems:
1. Identify key factual claims in the question
2. Cross-reference information for consistency
3. Be skeptical of unverified assumptions
4. Focus on concrete, verifiable details

IMPORTANT: Format your final answer on its own line starting with: ANSWER:

When reviewing other solutions: You are negative side. You disagree with the affirmative side's points. Provide
your reasons and answer""",
        "DebaterC": """You are a critical thinker participating in collaborative problem solving.
Your task is to identify potential flaws and strengthen arguments.

When solving problems:
1. Question assumptions others might take for granted
2. Consider alternative interpretations
3. Look for edge cases or exceptions
4. Play devil's advocate when appropriate

IMPORTANT: Format your final answer on its own line starting with: ANSWER:

When reviewing other solutions: You are affirmative side. Please express your viewpoints.""",
        "DebaterD": """You are a synthesis expert participating in collaborative problem solving.
Your task is to combine insights and find common ground.

When solving problems:
1. Look for patterns across different perspectives
2. Identify areas of agreement and disagreement
3. Synthesize the strongest elements from each view
4. Build consensus around the best answer

IMPORTANT: Format your final answer on its own line starting with: ANSWER:

When reviewing other solutions: You are negative side. You disagree with the affirmative side's points. Provide
your reasons and answers""",
    }

    # Run debate with custom prompts
    result = await run_debate_with_custom_prompts(
        task=prompt,
        model_client=model_client,
        domain_config=domain,
        judge=judge,
        evaluator=evaluator,
        system_prompts=system_prompts,
        num_debaters=num_debaters,
        max_rounds=max_rounds,
        topology_type=TopologyType.CIRCULAR,
        context={"expected_answer": expected_answer},
    )

    # Output
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Judge: {result.judge_mode.value}")
    if result.selected_debater_id:
        print(f"Selected: {result.selected_debater_id}")
    print(f"\nFINAL ANSWER: {result.final_solution}")

    if result.evaluation_result.details:
        f1 = result.evaluation_result.details.get("f1", 0)
        em = result.evaluation_result.details.get("exact_match", False)
        print(f"\nF1 Score: {f1:.4f}")
        print(f"Exact Match: {em}")

    if result.success:
        print("\nCORRECT!")
        print("ALL TESTS PASSED !#!#")
    else:
        print("\nINCORRECT")
        print(f"Expected: {expected_answer}")

    print(f"\nTime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
