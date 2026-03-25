"""
Multi-Agent Debate template for HumanEval benchmark.

Architecture:
    Debaters -> Judge (select OR synthesize) -> Evaluator (test) -> Result

Judge Modes:
    - DISCRIMINATIVE: Judge selects the best solution from candidates
    - EXTRACTIVE: Judge synthesizes a new solution combining best ideas
"""

import asyncio
import sys
import time

import yaml
from autogen_core.models import ChatCompletionClient

from agbench.agents.debating_agent import (
    CodingDomainConfig,
    CodeExecutionEvaluator,
    DiscriminativeJudge,
    ExtractiveJudge,
    JudgeMode,
    TopologyType,
    run_debate,
)


async def main() -> None:
    start_time = time.time()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open("prompt.txt", "r") as f:
        prompt = f.read().strip()

    with open("test.txt", "r") as f:
        test_code = f.read().strip()

    from custom_code_executor import ENTRY_POINT

    entry_point = ENTRY_POINT

    model_client = ChatCompletionClient.load_component(config["model_config"])

    # Configuration
    num_debaters = 4
    max_rounds = 2
    judge_mode = JudgeMode.DISCRIMINATIVE  # or EXTRACTIVE

    print(f"Multi-Agent Debate: {num_debaters} debaters, {max_rounds} rounds, {judge_mode.value} judge")
    print("=" * 60)

    # Setup
    domain = CodingDomainConfig(language="python")
    judge = (
        DiscriminativeJudge(model_client, domain)
        if judge_mode == JudgeMode.DISCRIMINATIVE
        else ExtractiveJudge(model_client, domain)
    )
    evaluator = CodeExecutionEvaluator(test_code=test_code, entry_point=entry_point)

    # Run debate
    result = await run_debate(
        task=prompt,
        model_client=model_client,
        domain=domain,
        judge=judge,
        evaluator=evaluator,
        num_debaters=num_debaters,
        max_rounds=max_rounds,
        topology_type=TopologyType.CIRCULAR,
        context={"test_code": test_code, "entry_point": entry_point},
    )

    # Output
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Judge: {result.judge_mode.value}")
    if result.selected_debater_id:
        print(f"Selected: {result.selected_debater_id}")
    print(f"\nSolution:\n```python\n{result.final_solution}\n```")

    if result.success:
        print("\nALL TESTS PASSED !#!#")
        print("TERMINATE")
    else:
        print("\nTESTS FAILED")

    print(f"\nTime: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
