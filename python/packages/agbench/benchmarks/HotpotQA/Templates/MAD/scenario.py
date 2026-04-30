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
from agbench.scenario_logging import emit_llm_call, emit_run_summary
from autogen_core import DefaultTopicId, SingleThreadedAgentRuntime, TypeSubscription
from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

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
from agbench.agents.debating_agent.protocol import (
    FinalDebaterResponse,
    JudgeResponse,
)


class HotpotQAExtractiveJudge(ExtractiveJudge):
    """Extractive judge tuned for HotpotQA short-phrase answers.

    HotpotQA expects answers like "1789", "yes", or "Marvel Cinematic Universe"
    — typically a few words. The base ExtractiveJudge prompt is too generic and
    invites verbose explanations that hurt F1/EM. This override emphasizes:
      - Synthesize a NEW answer (don't just pick from candidates).
      - Keep the answer to a short phrase / very few words.
    """

    async def judge(self, solutions: list[FinalDebaterResponse], task: DebateTask) -> JudgeResponse:
        if not solutions:
            raise ValueError("No solutions to judge")
        if len(solutions) == 1:
            return JudgeResponse(
                solution=solutions[0].solution,
                mode=self.mode,
                reasoning="Only one solution provided.",
                selected_debater_id=solutions[0].debater_id,
            )

        system = (
            "You are an expert judge for a HotpotQA short-answer question. "
            "Several debaters proposed candidate answers. Your job is to "
            "synthesize the FINAL ANSWER yourself — do not just pick one of "
            "the candidates verbatim. Combine the strongest evidence and "
            "reasoning across the debaters and produce your own answer.\n\n"
            "CRITICAL: HotpotQA answers are very short — usually a single "
            "phrase, name, year, number, or 'yes'/'no'. Do NOT write a "
            "sentence. Do NOT include explanation in the answer. Output the "
            "fewest words that fully answer the question.\n\n"
            "Output format (exactly):\n"
            "REASONING: <one or two sentences>\n"
            "SOLUTION: <a short answer, only the answer itself>"
        )

        # Include each debater's full reasoning (explanation) AND extracted
        # answer. The reasoning often disambiguates between debaters that
        # arrived at the same short phrase via different paths, and lets the
        # judge prefer the one with stronger evidence.
        user_parts = [f"Original question:\n{task.prompt}\n\nDebater submissions:\n"]
        for sol in solutions:
            reasoning = (sol.explanation or "").strip()
            if len(reasoning) > 2000:
                reasoning = reasoning[:2000] + "  ...[truncated]"
            user_parts.append(
                f"\n=== {sol.debater_id} ===\n"
                f"Reasoning:\n{reasoning}\n\n"
                f"Their final answer: {sol.solution}\n"
            )
        user_parts.append(
            "\nSynthesize the final HotpotQA answer using the strongest "
            "evidence across debaters. Remember: very few words, no sentence, "
            "no explanation in the SOLUTION line."
        )

        messages = [
            SystemMessage(content=system),
            UserMessage(content="".join(user_parts), source="user"),
        ]

        result = await self._model_client.create(messages)
        response = str(result.content) if not isinstance(result.content, str) else result.content

        # Prefer the SOLUTION: line; fall back to extract_solution / full text.
        m = re.search(r"SOLUTION:\s*(.+?)(?:\n\s*\n|\Z)", response, re.IGNORECASE | re.DOTALL)
        if m:
            solution = m.group(1).strip()
            # If the model wrote multiple lines, take the first non-empty one.
            for line in solution.splitlines():
                line = line.strip()
                if line and not line.lower().startswith("solution:"):
                    solution = line
                    break
        else:
            solution = self._domain_config.extract_solution(response)

        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:SOLUTION:|\Z)", response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return JudgeResponse(
            solution=solution,
            mode=self.mode,
            reasoning=reasoning,
            selected_debater_id=None,
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
    model_client: ChatCompletionClient | dict[str, ChatCompletionClient],
    domain_config: ReasoningDomainConfig,
    judge: DiscriminativeJudge | ExtractiveJudge,
    evaluator: SolutionEvaluator,
    system_prompts: dict[str, str],
    num_debaters: int = 4,
    max_rounds: int = 2,
    topology_type: TopologyType = TopologyType.CIRCULAR,
    context: dict[str, Any] | None = None,
):
    """Run debate with custom system prompts per debater.

    `model_client` can be a single client (used by all debaters) or a dict
    mapping debater_id -> client (e.g. {"DebaterA": qwen_client, "DebaterB": gemma_client}).
    """
    context = context or {}

    # Setup topology
    topology = TopologyConfig(topology_type=topology_type, num_debaters=num_debaters)
    debater_ids = topology.get_debater_ids()
    connections = topology.get_connections()

    def _client_for(did: str) -> ChatCompletionClient:
        if isinstance(model_client, dict):
            if did not in model_client:
                raise ValueError(f"No model_client provided for debater '{did}'")
            return model_client[did]
        return model_client

    # Create runtime
    runtime = SingleThreadedAgentRuntime()

    # Register debaters with custom system prompts
    for debater_id in debater_ids:
        num_neighbors = topology.get_num_neighbors(debater_id)
        system_prompt = system_prompts.get(debater_id)  # None uses domain default
        debater_client = _client_for(debater_id)

        await Debater.register(
            runtime,
            debater_id,
            lambda did=debater_id, nn=num_neighbors, sp=system_prompt, mc=debater_client: Debater(
                model_client=mc,
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

    # Configuration
    num_debaters = 4
    max_rounds = 3
    judge_mode = JudgeMode.EXTRACTIVE

    # Resolve debater_id -> client mapping. Supports two layouts:
    #   1. Single `model_config` (legacy): one client used for all debaters + judge.
    #   2. Per-debater `model_config_debaterA` / `_debaterB` / ... and an
    #      optional `model_config_judge`. Falls back to `model_config` for any
    #      role not explicitly listed.
    topology = TopologyConfig(topology_type=TopologyType.CIRCULAR, num_debaters=num_debaters)
    debater_ids = topology.get_debater_ids()

    debater_clients: dict[str, ChatCompletionClient] = {}
    default_section = config.get("model_config")
    for did in debater_ids:
        section_key = f"model_config_debater{did[-1]}"  # DebaterA -> model_config_debaterA
        section = config.get(section_key, default_section)
        if section is None:
            raise ValueError(f"No model config found for {did} ({section_key} or model_config)")
        debater_clients[did] = ChatCompletionClient.load_component(section)

    judge_section = config.get("model_config_judge", default_section)
    if judge_section is None:
        raise ValueError("No model config found for judge (model_config_judge or model_config)")
    judge_client = ChatCompletionClient.load_component(judge_section)

    # === Latency instrumentation ===========================================
    # Per-call markers (parsed by reports/csv tooling, mirrors the convention
    # used by SelectorGroupChat scenario):
    #   [LATENCY] role=debater label=DebaterA round=N duration_s=... prompt_tokens=... completion_tokens=...
    #   [LATENCY] role=judge   label=judge        duration_s=... prompt_tokens=... completion_tokens=...
    # Plus a per-round wall-clock entry, fired when the last debater of round N
    # returns from its LLM call:
    #   [ROUND] round=N wallclock_s=... debaters=N
    # wallclock_s = max(end_time over debaters in round N) - min(start_time over them).
    round_state: dict[int, dict[str, float]] = {}
    # Global token accumulator across debater + judge LLM calls. The MAD
    # runtime is single-threaded (SingleThreadedAgentRuntime), but multiple
    # debater coroutines can be in-flight on the asyncio event loop within a
    # round, so we guard the accumulator with an asyncio.Lock. The lock is
    # held only across plain ints/dict ops (no awaits inside) so contention
    # is negligible.
    global_usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
    rounds_completed = {"n": 0}
    usage_lock = asyncio.Lock()

    async def _add_global_usage(usage) -> None:
        if usage is None:
            return
        async with usage_lock:
            global_usage["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
            global_usage["completion_tokens"] += int(getattr(usage, "completion_tokens", 0) or 0)
            global_usage["reasoning_tokens"] += int(getattr(usage, "reasoning_tokens", 0) or 0)

    def _instrument_debater(client: ChatCompletionClient, debater_id: str) -> None:
        _orig = client.create
        call_idx = {"n": 0}  # round number = nth call (each debater calls model once per round)

        async def _timed(*args, **kwargs):
            round_n = call_idx["n"]
            call_idx["n"] += 1
            t0 = time.perf_counter()
            rs = round_state.setdefault(round_n, {"start": t0, "end": t0, "started": 0, "finished": 0})
            if rs["started"] == 0:
                rs["start"] = t0
            rs["started"] += 1

            result = None
            try:
                result = await _orig(*args, **kwargs)
                return result
            finally:
                t1 = time.perf_counter()
                dt = t1 - t0
                usage = getattr(result, "usage", None) if result is not None else None
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                rt = getattr(usage, "reasoning_tokens", None) if usage else None
                print(
                    f"[LATENCY] role=debater label={debater_id} round={round_n} "
                    f"duration_s={dt:.3f} prompt_tokens={pt} completion_tokens={ct}",
                    flush=True,
                )
                emit_llm_call(
                    agent=debater_id,
                    duration_s=dt,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    reasoning_tokens=rt,
                    role="debater",
                    round_idx=round_n,
                )
                await _add_global_usage(usage)
                rs["end"] = max(rs["end"], t1)
                rs["finished"] += 1
                if rs["finished"] >= num_debaters:
                    wc = rs["end"] - rs["start"]
                    print(
                        f"[ROUND] round={round_n} wallclock_s={wc:.3f} debaters={num_debaters}",
                        flush=True,
                    )
                    # Track the highest completed debate round (1-indexed).
                    rounds_completed["n"] = max(rounds_completed["n"], round_n + 1)

        client.create = _timed

    def _instrument_judge(client: ChatCompletionClient) -> None:
        _orig = client.create

        async def _timed(*args, **kwargs):
            t0 = time.perf_counter()
            result = None
            try:
                result = await _orig(*args, **kwargs)
                return result
            finally:
                dt = time.perf_counter() - t0
                usage = getattr(result, "usage", None) if result is not None else None
                pt = getattr(usage, "prompt_tokens", None) if usage else None
                ct = getattr(usage, "completion_tokens", None) if usage else None
                rt = getattr(usage, "reasoning_tokens", None) if usage else None
                print(
                    f"[LATENCY] role=judge label=judge duration_s={dt:.3f} prompt_tokens={pt} completion_tokens={ct}",
                    flush=True,
                )
                emit_llm_call(
                    agent="judge",
                    duration_s=dt,
                    prompt_tokens=pt,
                    completion_tokens=ct,
                    reasoning_tokens=rt,
                    role="judge",
                )
                await _add_global_usage(usage)

        client.create = _timed

    for did, client in debater_clients.items():
        _instrument_debater(client, did)
    _instrument_judge(judge_client)
    # =======================================================================

    print(f"Multi-Agent Debate: {num_debaters} debaters, {max_rounds} rounds, {judge_mode.value} judge")
    for did, client in debater_clients.items():
        print(f"  {did}: {getattr(client, '_resolved_model', getattr(client, 'model', '?'))}")
    print(f"  Judge: {getattr(judge_client, '_resolved_model', getattr(judge_client, 'model', '?'))}")
    print("=" * 60)

    # Setup domain and evaluator
    domain = ReasoningDomainConfig()
    if judge_mode == JudgeMode.DISCRIMINATIVE:
        judge = DiscriminativeJudge(judge_client, domain)
    else:
        # Use HotpotQA-specific extractive judge so the synthesized answer is a
        # short phrase (matches HotpotQA's F1/EM scoring expectations).
        judge = HotpotQAExtractiveJudge(judge_client, domain)
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
        "DebaterD": """You are a synthesis-oriented reasoner participating in collaborative problem solving.
Your task is to weigh competing claims and find the best-supported answer.

When solving problems:
1. Map out what each piece of evidence implies
2. Reconcile apparent contradictions when possible
3. Pick the answer that the evidence most directly supports
4. Avoid speculation beyond what the evidence shows

IMPORTANT: Format your final answer on its own line starting with: ANSWER:

When reviewing other solutions: You are negative side. You disagree with the affirmative side's points.
Provide your reasons and answer.""",
    }

    # Run debate with custom prompts
    result = await run_debate_with_custom_prompts(
        task=prompt,
        model_client=debater_clients,
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

    # Canonical end-of-run summary parsed by run_cmd into result.json.
    # A "round" in MAD is one debate round in which every debater contributes;
    # we report the highest completed round number (1-indexed). The judge
    # call happens after the rounds and is aggregated into the token totals
    # but does not increment the round counter.
    class _Usage:
        def __init__(self, p, c, r):
            self.prompt_tokens = p
            self.completion_tokens = c
            self.reasoning_tokens = r

    emit_run_summary(
        [
            _Usage(
                global_usage["prompt_tokens"],
                global_usage["completion_tokens"],
                global_usage["reasoning_tokens"],
            )
        ],
        rounds=rounds_completed["n"],
    )


if __name__ == "__main__":
    asyncio.run(main())
