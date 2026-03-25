"""
Example: Multi-Agent Debate for Question Answering

This example demonstrates how to use the debating agent framework
for reasoning and QA tasks through collaborative debate.

Usage:
    python -m agbench.agents.debating_agent.example_qa
"""

import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient

from . import (
    DiscriminativeJudge,
    ExactMatchEvaluator,
    ReasoningDomainConfig,
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

    # For OpenAI:
    # model_client = OpenAIChatCompletionClient(
    #     model="gpt-4",
    #     api_key="sk-...",
    # )

    # QA problem - reasoning task
    question = """
    Alice, Bob, and Carol are standing in a line. Alice is not first.
    Bob is not last. Carol is not next to Alice.

    In what order are they standing (from first to last)?
    """

    expected_answer = "Bob, Carol, Alice"

    # Setup domain and evaluator
    domain = ReasoningDomainConfig()
    judge = DiscriminativeJudge(model_client, domain)
    evaluator = ExactMatchEvaluator(expected_answer=expected_answer, case_sensitive=False)

    print("QA Multi-Agent Debate")
    print("=" * 60)
    print(f"Question: {question.strip()}")
    print(f"Expected answer: {expected_answer}")
    print("=" * 60)

    # Run debate with 4 debaters, 2 rounds
    result = await run_debate(
        task=question,
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

    # Show all debater answers
    print("\n--- All Debater Answers ---")
    for sol in result.all_solutions:
        print(f"{sol.debater_id}: {sol.solution}")


if __name__ == "__main__":
    asyncio.run(main())
