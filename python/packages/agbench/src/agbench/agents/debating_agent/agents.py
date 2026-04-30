"""
Agent implementations for multi-agent debate.
"""

import asyncio

from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from .protocol import (
    DebateResult,
    DebateTask,
    DebaterRequest,
    DomainConfig,
    FinalDebaterResponse,
    IntermediateDebaterResponse,
    Judge,
    SolutionEvaluator,
)


@default_subscription
class Debater(RoutedAgent):
    """Debater agent that generates and refines solutions through debate."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        domain_config: DomainConfig,
        topic_type: str,
        num_neighbors: int,
        max_rounds: int,
        debater_id: str,
        system_prompt: str | None = None,
    ) -> None:
        super().__init__(f"Debater {debater_id}")
        self._model_client = model_client
        self._domain_config = domain_config
        self._topic_type = topic_type
        self._num_neighbors = num_neighbors
        self._max_rounds = max_rounds
        self._debater_id = debater_id
        self._history: list[LLMMessage] = []
        self._buffer: dict[int, list[IntermediateDebaterResponse]] = {}
        self._round = 0
        self._done = False  # set to True once a Final has been published (success or error)
        # Cached on first message arrival so the error-recovery path can build
        # placeholder Intermediates for rounds we couldn't complete.
        self._latest_task: "DebateTask | None" = None
        # Serialize handle_request invocations on this debater. autogen-core's
        # SingleThreadedAgentRuntime does NOT serialize per-agent message
        # handlers — concurrent handler tasks can interleave at await points.
        # Without this lock, handle_response (triggered by neighbor
        # Intermediates) can fire send_message → handle_request for round N+1
        # while round N's `await client.create(...)` is still pending, causing
        # _history to be mutated mid-call and producing requests like
        # [system, user, user] that strict chat templates (gemma-3, Mistral,
        # ...) reject with "Conversation roles must alternate".
        self._request_lock = asyncio.Lock()

        # Use custom system prompt if provided, otherwise use domain default
        prompt = system_prompt if system_prompt is not None else domain_config.get_system_prompt()
        self._system_messages = [SystemMessage(content=prompt)]

    async def _publish_error_final(self, exc: BaseException, round_at_failure: int) -> None:
        """Emit placeholder Intermediates for the rounds we still owe neighbors,
        then a placeholder FinalDebaterResponse so the Aggregator can reach
        quorum (`len(buffer) >= num_debaters`) when this debater fails.

        Why the Intermediates matter: in CIRCULAR (and other neighbor-based)
        topologies, each handle_response only fires when buffer[round] reaches
        num_neighbors. If we error out and never publish Intermediates for the
        rounds we haven't completed, our neighbors deadlock waiting for input
        that never arrives — they never call their next round and never
        publish their own Finals. The Aggregator then sits idle and the task
        ends with "No result collected".

        round_at_failure = self._round at the time of failure (BEFORE the
        increment that didn't run). With max_rounds=R, intermediate rounds
        published on success are 1..R-1. Rounds we still owe neighbors are
        range(round_at_failure + 1, R).
        """
        err_msg = f"<{type(exc).__name__}: {exc}>"
        print(f"[{self._debater_id}] ERROR in round {round_at_failure}: {err_msg}", flush=True)
        self._done = True

        # Placeholder Intermediates for all unfinished rounds. Skip if we don't
        # have a task reference (shouldn't happen — handle_request sets it
        # before the try block — but defensive).
        if self._latest_task is not None:
            for r in range(round_at_failure + 1, self._max_rounds):
                await self.publish_message(
                    IntermediateDebaterResponse(
                        solution="",
                        explanation=err_msg,
                        original_task=self._latest_task,
                        round=r,
                        debater_id=self._debater_id,
                        metadata={"failed": True, "placeholder": True},
                    ),
                    topic_id=DefaultTopicId(type=self._topic_type),
                )

        await self.publish_message(
            FinalDebaterResponse(
                solution="",
                debater_id=self._debater_id,
                explanation=err_msg,
                confidence=0.0,
                metadata={"failed": True, "round": round_at_failure},
            ),
            topic_id=DefaultTopicId(),
        )

    @message_handler
    async def handle_request(self, message: DebaterRequest, ctx: MessageContext) -> None:
        if self._done:
            return  # already published a Final (success or error)

        async with self._request_lock:
            # Re-check after acquiring the lock — _done may have flipped while
            # we were queued behind a prior invocation that errored out.
            if self._done:
                return

            # Cache the task reference for the error-recovery path (placeholder
            # Intermediates need original_task to be valid IntermediateDebaterResponses).
            self._latest_task = message.original_task

            print(f"\n{'='*60}")
            print(f"[{self._debater_id}] Round {self._round + 1}/{self._max_rounds}")
            print(f"{'='*60}")

            try:
                self._history.append(UserMessage(content=message.content, source="user"))
                result = await self._model_client.create(self._system_messages + self._history)
                response = str(result.content) if not isinstance(result.content, str) else result.content
                self._history.append(AssistantMessage(content=response, source=self._debater_id))

                solution = self._domain_config.extract_solution(response)
                self._round += 1
            except Exception as e:
                await self._publish_error_final(e, self._round)
                return

            print(f"[{self._debater_id}] Response:")
            print(f"{'-'*40}")
            print(response[:2000] + ("..." if len(response) > 2000 else ""))
            print(f"{'-'*40}")
            print(f"[{self._debater_id}] Extracted solution:")
            print(solution[:1000] + ("..." if len(solution) > 1000 else ""))
            print()

            if self._round >= self._max_rounds:
                self._done = True
                await self.publish_message(
                    FinalDebaterResponse(solution=solution, debater_id=self._debater_id, explanation=response),
                    topic_id=DefaultTopicId(),
                )
            else:
                await self.publish_message(
                    IntermediateDebaterResponse(
                        solution=solution,
                        explanation=response,
                        original_task=message.original_task,
                        round=self._round,
                        debater_id=self._debater_id,
                    ),
                    topic_id=DefaultTopicId(type=self._topic_type),
                )

    @message_handler
    async def handle_response(self, message: IntermediateDebaterResponse, ctx: MessageContext) -> None:
        if self._done:
            return  # ignore neighbor traffic after we've published our Final

        # Also cache here so we can recover even if our own handle_request never ran.
        self._latest_task = message.original_task

        self._buffer.setdefault(message.round, []).append(message)

        if len(self._buffer[message.round]) >= self._num_neighbors:
            # Build refinement prompt
            # Decide whether to truncate neighbor reasoning. Default is to
            # pass the full explanation through; only truncate when the
            # cumulative length across all neighbors would exceed the
            # domain's per-prompt budget. When triggered, distribute the
            # budget evenly across neighbors and log it so the degradation
            # is visible in console_log.txt.
            neighbors = self._buffer[message.round]
            total_chars = sum(
                len((r.explanation or "").strip()) for r in neighbors
            )
            budget = getattr(
                self._domain_config, "SHARED_EXPLANATION_BUDGET_CHARS", None
            )
            per_neighbor_cap = None
            if budget is not None and total_chars > budget and len(neighbors) > 0:
                per_neighbor_cap = budget // len(neighbors)
                print(
                    f"[{self._debater_id}] [TRUNCATE] round={message.round + 1} "
                    f"cumulative_neighbor_chars={total_chars} budget={budget} "
                    f"neighbors={len(neighbors)} per_neighbor_cap={per_neighbor_cap}",
                    flush=True,
                )

            neighbor_solutions = "\n".join(
                self._domain_config.format_solution_for_sharing(
                    r.solution, r.explanation, r.debater_id, max_chars=per_neighbor_cap
                )
                for r in neighbors
            )
            prompt = self._domain_config.get_refinement_prompt_template().format(
                neighbor_solutions=neighbor_solutions,
                original_prompt=message.original_task.prompt,
                round=message.round + 1,
            )
            self._buffer.pop(message.round)
            try:
                await self.send_message(
                    DebaterRequest(content=prompt, original_task=message.original_task, round=message.round),
                    self.id,
                )
            except Exception as e:
                # Refinement send_message itself failed (rare — usually only if
                # the recipient agent is gone). Treat as terminal for this debater.
                await self._publish_error_final(e, message.round)


@default_subscription
class DebateAggregator(RoutedAgent):
    """Aggregator that orchestrates debate, judgment, and evaluation."""

    def __init__(
        self,
        domain_config: DomainConfig,
        judge: Judge,
        evaluator: SolutionEvaluator,
        num_debaters: int,
    ) -> None:
        super().__init__("Aggregator")
        self._domain_config = domain_config
        self._judge = judge
        self._evaluator = evaluator
        self._num_debaters = num_debaters
        self._buffer: list[FinalDebaterResponse] = []
        self._current_task: DebateTask | None = None

    @message_handler
    async def handle_task(self, message: DebateTask, ctx: MessageContext) -> None:
        print(f"[Aggregator] Received task: {message.task_id}")
        self._current_task = message
        self._buffer.clear()

        context_str = "\n".join(
            f"{k}: {v}" for k, v in message.context.items() if not isinstance(v, str) or "\n" not in v
        )
        initial_prompt = self._domain_config.get_initial_prompt_template().format(
            prompt=message.prompt,
            context=context_str,
        )

        await self.publish_message(
            DebaterRequest(content=initial_prompt, original_task=message, round=0),
            topic_id=DefaultTopicId(),
        )

    @message_handler
    async def handle_final_response(self, message: FinalDebaterResponse, ctx: MessageContext) -> None:
        print(f"\n[Aggregator] Received final response from {message.debater_id}")
        print(f"  Solution preview: {message.solution[:200]}..." if len(message.solution) > 200 else f"  Solution: {message.solution}")
        self._buffer.append(message)

        if len(self._buffer) >= self._num_debaters:
            if self._current_task is None:
                raise RuntimeError("No current task")

            print(f"\n{'='*60}")
            print(f"[Aggregator] All {self._num_debaters} solutions received")
            print(f"{'='*60}")
            for i, sol in enumerate(self._buffer):
                print(f"\n--- {sol.debater_id} ---")
                print(sol.solution[:500] + ("..." if len(sol.solution) > 500 else ""))

            # Judge
            print(f"\n{'='*60}")
            print(f"[Aggregator] Judging ({self._judge.mode.value})")
            print(f"{'='*60}")
            judge_response = await self._judge.judge(self._buffer, self._current_task)
            print(f"[Aggregator] Judge selected: {judge_response.selected_debater_id}")
            print(f"[Aggregator] Judge reasoning: {judge_response.reasoning[:500]}..." if len(judge_response.reasoning) > 500 else f"[Aggregator] Judge reasoning: {judge_response.reasoning}")

            # Evaluate
            print(f"\n{'='*60}")
            print("[Aggregator] Evaluating solution")
            print(f"{'='*60}")
            eval_result = await self._evaluator.evaluate(judge_response.solution, self._current_task)
            print(f"[Aggregator] Evaluation result: passed={eval_result.passed}, score={eval_result.score}")
            if eval_result.error:
                print(f"[Aggregator] Evaluation error: {eval_result.error}")

            # Publish result
            result = DebateResult(
                final_solution=judge_response.solution,
                judge_mode=judge_response.mode,
                judge_reasoning=judge_response.reasoning,
                selected_debater_id=judge_response.selected_debater_id,
                all_solutions=list(self._buffer),
                evaluation_result=eval_result,
                success=eval_result.passed,
            )
            await self.publish_message(result, topic_id=DefaultTopicId())
            self._buffer.clear()


@default_subscription
class ResultCollector(RoutedAgent):
    """Collects the final debate result."""

    def __init__(self) -> None:
        super().__init__("ResultCollector")
        self.result: DebateResult | None = None

    @message_handler
    async def handle_result(self, message: DebateResult, ctx: MessageContext) -> None:
        print(f"\n{'='*60}")
        print(f"[ResultCollector] FINAL RESULT")
        print(f"{'='*60}")
        print(f"  Success: {message.success}")
        print(f"  Judge mode: {message.judge_mode.value}")
        print(f"  Selected debater: {message.selected_debater_id}")
        print(f"  Final solution:")
        print(f"{'-'*40}")
        print(message.final_solution)
        print(f"{'='*60}\n")
        self.result = message
