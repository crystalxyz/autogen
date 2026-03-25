"""
Example: Multi-Agent Debate for Code Generation

This example demonstrates how to use the debating agent framework
to solve coding problems through collaborative debate.

Usage:
    python -m agbench.agents.debating_agent.example.example_code
"""

import asyncio

from autogen_ext.models.openai import OpenAIChatCompletionClient

from .. import (
    CodeExecutionEvaluator,
    CodingDomainConfig,
    DiscriminativeJudge,
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

    # Coding problem - implement a function
    prompt = """
    Implement a function `is_palindrome(s: str) -> bool` that checks if a string
    is a palindrome. The function should ignore case and non-alphanumeric characters.

    Examples:
        is_palindrome("A man, a plan, a canal: Panama") -> True
        is_palindrome("race a car") -> False
        is_palindrome("") -> True
    """

    # Test code for evaluation
    test_code = """
def check(candidate):
    assert candidate("A man, a plan, a canal: Panama") == True
    assert candidate("race a car") == False
    assert candidate("") == True
    assert candidate("a") == True
    assert candidate("ab") == False
    assert candidate("aba") == True
    assert candidate("Able was I ere I saw Elba") == True
    assert candidate("No 'x' in Nixon") == True
    print("All tests passed!")

check(is_palindrome)
"""

    entry_point = "is_palindrome"

    # Setup domain and evaluator
    domain = CodingDomainConfig(language="python")
    judge = DiscriminativeJudge(model_client, domain)
    evaluator = CodeExecutionEvaluator(test_code=test_code, entry_point=entry_point)

    print("Code Generation Multi-Agent Debate")
    print("=" * 60)
    print(f"Problem: {prompt.strip()}")
    print("=" * 60)

    # Run debate with 4 debaters, 2 rounds
    result = await run_debate(
        task=prompt,
        model_client=model_client,
        domain=domain,
        judge=judge,
        evaluator=evaluator,
        num_debaters=4,
        max_rounds=2,
        topology_type=TopologyType.CIRCULAR,
        context={"test_code": test_code, "entry_point": entry_point},
    )

    # Output results
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Judge mode: {result.judge_mode.value}")
    if result.selected_debater_id:
        print(f"Selected debater: {result.selected_debater_id}")
    print(f"\nGenerated code:\n```python\n{result.final_solution}\n```")
    print(f"\nAll tests passed: {result.success}")

    if result.evaluation_result.details:
        print(f"Details: {result.evaluation_result.details}")

    # Show all debater solutions
    print("\n--- All Debater Solutions ---")
    for sol in result.all_solutions:
        print(f"\n{sol.debater_id}:")
        print(f"```python\n{sol.solution}\n```")


if __name__ == "__main__":
    asyncio.run(main())
