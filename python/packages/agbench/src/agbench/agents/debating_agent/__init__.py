"""
Generic Multi-Agent Debating Framework

Architecture:
    Debaters -> Judge (select OR synthesize) -> Evaluator (test) -> Result

Usage:
    from agbench.agents.debating_agent import run_debate, JudgeMode

    result = await run_debate(
        task="def is_prime(n): ...",
        model_client=client,
        domain="coding",
        judge_mode=JudgeMode.DISCRIMINATIVE,  # or EXTRACTIVE
    )
"""

from typing import Any

from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.models import ChatCompletionClient

from .protocol import (
    DebateResult,
    DebateTask,
    DebaterRequest,
    DomainConfig,
    EvaluationResult,
    FinalDebaterResponse,
    IntermediateDebaterResponse,
    Judge,
    JudgeMode,
    JudgeResponse,
    SolutionEvaluator,
)
from .topology import TopologyConfig, TopologyType
from .domains import CodingDomainConfig, MathDomainConfig, ReasoningDomainConfig, get_domain_config
from .judge import DiscriminativeJudge, ExtractiveJudge, SimpleVotingJudge
from .evaluators import CodeExecutionEvaluator, MathEvaluator, ExactMatchEvaluator, PassThroughEvaluator
from .agents import Debater, DebateAggregator, ResultCollector

__all__ = [
    "run_debate",
    "JudgeMode",
    "DebateTask",
    "DebateResult",
    "DomainConfig",
    "Judge",
    "SolutionEvaluator",
    "TopologyConfig",
    "TopologyType",
    "CodingDomainConfig",
    "MathDomainConfig",
    "ReasoningDomainConfig",
    "DiscriminativeJudge",
    "ExtractiveJudge",
    "SimpleVotingJudge",
    "CodeExecutionEvaluator",
    "MathEvaluator",
    "ExactMatchEvaluator",
    "PassThroughEvaluator",
    "Debater",
    "DebateAggregator",
    "ResultCollector",
]


async def run_debate(
    task: str,
    model_client: ChatCompletionClient | list[ChatCompletionClient] | dict[str, ChatCompletionClient],
    domain: str | DomainConfig = "coding",
    judge: Judge = None,
    evaluator: SolutionEvaluator = None,
    num_debaters: int = 4,
    max_rounds: int = 2,
    topology_type: TopologyType = TopologyType.CIRCULAR,
    task_id: str = "task_001",
    context: dict[str, Any] | None = None,
) -> DebateResult:
    """Run a multi-agent debate.

    Args:
        task: The problem description
        model_client: LLM client(s) for debaters. Can be:
            - Single ChatCompletionClient: used for all debaters
            - List of ChatCompletionClient: one per debater (must match num_debaters)
            - Dict mapping debater_id to ChatCompletionClient
        domain: "coding", "math", "reasoning" or DomainConfig instance
        judge: Judge instance (required)
        evaluator: Evaluator instance (required)
        num_debaters: Number of debater agents
        max_rounds: Number of debate rounds
        topology_type: How debaters connect (CIRCULAR, GRID_2D, etc.)
        task_id: Task identifier
        context: Additional context (e.g., test_code, entry_point)

    Returns:
        DebateResult with final_solution, success, etc.
    """
    if judge is None:
        raise ValueError("judge is required")
    if evaluator is None:
        raise ValueError("evaluator is required")

    # Get domain config
    domain_config = get_domain_config(domain) if isinstance(domain, str) else domain
    context = context or {}

    # Setup topology
    topology = TopologyConfig(topology_type=topology_type, num_debaters=num_debaters)
    debater_ids = topology.get_debater_ids()
    connections = topology.get_connections()

    # Build model client mapping
    def get_client_for_debater(debater_id: str, index: int) -> ChatCompletionClient:
        if isinstance(model_client, dict):
            if debater_id not in model_client:
                raise ValueError(f"No model_client provided for debater '{debater_id}'")
            return model_client[debater_id]
        elif isinstance(model_client, list):
            if len(model_client) != num_debaters:
                raise ValueError(f"model_client list length ({len(model_client)}) must match num_debaters ({num_debaters})")
            return model_client[index]
        else:
            return model_client

    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    # Register debaters
    for idx, debater_id in enumerate(debater_ids):
        num_neighbors = topology.get_num_neighbors(debater_id)
        client = get_client_for_debater(debater_id, idx)
        await Debater.register(
            runtime,
            debater_id,
            lambda did=debater_id, nn=num_neighbors, mc=client: Debater(
                model_client=mc,
                domain_config=domain_config,
                topic_type=did,
                num_neighbors=nn,
                max_rounds=max_rounds,
                debater_id=did,
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
        DebateTask(task_id=task_id, prompt=task, context=context),
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
