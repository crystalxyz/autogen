"""
Judge implementations for multi-agent debate.
"""

import re

from autogen_core.models import ChatCompletionClient, SystemMessage, UserMessage

from .protocol import (
    DebateTask,
    DomainConfig,
    FinalDebaterResponse,
    Judge,
    JudgeMode,
    JudgeResponse,
)


class DiscriminativeJudge(Judge):
    """Judge that selects the best solution from candidates."""

    def __init__(self, model_client: ChatCompletionClient, domain_config: DomainConfig) -> None:
        self._model_client = model_client
        self._domain_config = domain_config

    @property
    def mode(self) -> JudgeMode:
        return JudgeMode.DISCRIMINATIVE

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

        # Build prompt
        system = f"""You are an expert judge evaluating {self._domain_config.name} solutions.
Analyze each solution carefully and select the BEST one.
Do not modify solutions - just select one.

Output format:
SELECTED: <debater_id>
REASONING: <your explanation>"""

        user_parts = [f"Original task:\n{task.prompt}\n\nCandidate solutions:\n"]
        for sol in solutions:
            user_parts.append(f"\n=== {sol.debater_id} ===\n{sol.solution}\n")
        user_parts.append("\nSelect the best solution.")

        messages = [
            SystemMessage(content=system),
            UserMessage(content="".join(user_parts), source="user"),
        ]

        result = await self._model_client.create(messages)
        response = str(result.content) if not isinstance(result.content, str) else result.content

        # Parse selection
        selected_id, reasoning = self._parse_selection(response, solutions)
        selected = next((s for s in solutions if s.debater_id == selected_id), solutions[0])

        return JudgeResponse(
            solution=selected.solution,
            mode=self.mode,
            reasoning=reasoning,
            selected_debater_id=selected.debater_id,
        )

    def _parse_selection(self, response: str, solutions: list[FinalDebaterResponse]) -> tuple[str, str]:
        valid_ids = [s.debater_id for s in solutions]

        match = re.search(r"SELECTED:\s*(\w+)", response, re.IGNORECASE)
        if match and match.group(1) in valid_ids:
            reasoning_match = re.search(r"REASONING:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else response
            return match.group(1), reasoning

        for sol in solutions:
            if sol.debater_id in response:
                return sol.debater_id, response

        return solutions[0].debater_id, response


class ExtractiveJudge(Judge):
    """Judge that synthesizes a new solution from candidates."""

    def __init__(self, model_client: ChatCompletionClient, domain_config: DomainConfig) -> None:
        self._model_client = model_client
        self._domain_config = domain_config

    @property
    def mode(self) -> JudgeMode:
        return JudgeMode.EXTRACTIVE

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

        # Build prompt
        system = f"""You are an expert judge synthesizing the best {self._domain_config.name} solution.
Analyze the candidate solutions and create an IMPROVED and more certain solution.

Output format:
REASONING: <your analysis>

SOLUTION:
<your synthesized solution>"""

        user_parts = [f"Original task:\n{task.prompt}\n\nCandidate solutions:\n"]
        for sol in solutions:
            user_parts.append(f"\n=== {sol.debater_id} ===\n{sol.solution}\n")
        user_parts.append("\nSynthesize an improved solution.")

        messages = [
            SystemMessage(content=system),
            UserMessage(content="".join(user_parts), source="user"),
        ]

        result = await self._model_client.create(messages)
        response = str(result.content) if not isinstance(result.content, str) else result.content

        # Extract solution and reasoning
        solution = self._domain_config.extract_solution(response)
        reasoning_match = re.search(r"REASONING:\s*(.+?)(?:SOLUTION:|```)", response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        return JudgeResponse(
            solution=solution,
            mode=self.mode,
            reasoning=reasoning,
            selected_debater_id=None,
        )


class SimpleVotingJudge(Judge):
    """Judge that uses majority voting (no LLM)."""

    def __init__(self, normalizer=None) -> None:
        self._normalizer = normalizer or (lambda x: x.strip())

    @property
    def mode(self) -> JudgeMode:
        return JudgeMode.DISCRIMINATIVE

    async def judge(self, solutions: list[FinalDebaterResponse], task: DebateTask) -> JudgeResponse:
        if not solutions:
            raise ValueError("No solutions to judge")

        # Group by normalized form
        groups: dict[str, list[FinalDebaterResponse]] = {}
        for sol in solutions:
            key = self._normalizer(sol.solution)
            groups.setdefault(key, []).append(sol)

        # Find most common
        winner_group = max(groups.values(), key=len)
        winner = winner_group[0]

        return JudgeResponse(
            solution=winner.solution,
            mode=self.mode,
            reasoning=f"Selected by majority vote ({len(winner_group)}/{len(solutions)} votes)",
            selected_debater_id=winner.debater_id,
        )
