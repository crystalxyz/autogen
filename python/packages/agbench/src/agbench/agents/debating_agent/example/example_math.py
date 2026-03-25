"""
Example: Multi-Agent Debate for Math Problems

This example demonstrates how to use the debating agent framework
to solve mathematical problems through collaborative debate.

Usage:
    python -m agbench.agents.debating_agent.example_math
"""

import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient

from . import (
    DiscriminativeJudge,
    MathDomainConfig,
    MathEvaluator,
    TopologyType,
    run_debate,
)


async def main() -> None:
    # Configure model client
    # For local endpoint (e.g., SGLang, vLLM):
    model_client = OpenAIChatCompletionClient(
        model="your-model-name",
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Math problem to solve
    problem = """
    A store sells apples for $2 each and oranges for $3 each.
    If a customer buys 5 apples and 4 oranges, and pays with a $50 bill,
    how much change should they receive?
    """

    expected_answer = 28  # 50 - (5*2 + 4*3) = 50 - 22 = 28

    # Setup domain and evaluator
    domain = MathDomainConfig(answer_format="{{answer}}")
    judge = DiscriminativeJudge(model_client, domain)
    evaluator = MathEvaluator(expected_answer=expected_answer)

    print("Math Multi-Agent Debate")
    print("=" * 60)
    print(f"Problem: {problem.strip()}")
    print(f"Expected answer: {expected_answer}")
    print("=" * 60)

    # Run debate with 4 debaters, 2 rounds
    result = await run_debate(
        task=problem,
        model_client=model_client,
        domain=domain,
        judge=judge,
        evaluator=evaluator,
        num_debaters=4,
        max_rounds=2,
        topology_type=TopologyType.CIRCULAR,
        context={"expected_answer": expected_answer},
    )

    # Output results
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Judge mode: {result.judge_mode.value}")
    if result.selected_debater_id:
        print(f"Selected debater: {result.selected_debater_id}")
    print(f"\nFinal answer: {result.final_solution}")
    print(f"Correct: {result.success}")

    if result.evaluation_result.details:
        print(f"Details: {result.evaluation_result.details}")


if __name__ == "__main__":
    asyncio.run(main())
