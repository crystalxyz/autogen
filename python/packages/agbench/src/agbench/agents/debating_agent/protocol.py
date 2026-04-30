"""
Message protocol and base interfaces for multi-agent debate.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JudgeMode(Enum):
    """Mode of operation for the judge."""
    DISCRIMINATIVE = "discriminative"  # Select one solution
    EXTRACTIVE = "extractive"          # Synthesize new solution


class DebateTask(BaseModel):
    """Initial task sent to all debaters."""
    task_id: str
    prompt: str
    context: dict[str, Any] = Field(default_factory=dict)


class DebaterRequest(BaseModel):
    """Request for a debater to generate or refine a solution."""
    content: str
    original_task: DebateTask
    round: int = 0


class IntermediateDebaterResponse(BaseModel):
    """Intermediate response shared between neighboring debaters."""
    solution: str
    explanation: str
    original_task: DebateTask
    round: int
    debater_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinalDebaterResponse(BaseModel):
    """Final solution from a debater after all debate rounds."""
    solution: str
    debater_id: str
    confidence: float = 1.0
    explanation: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class JudgeResponse(BaseModel):
    """Response from a judge after aggregating solutions."""
    solution: str
    mode: JudgeMode
    reasoning: str = ""
    selected_debater_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationResult(BaseModel):
    """Result from evaluating a solution."""
    passed: bool
    score: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class DebateResult(BaseModel):
    """Final result after debate, judgment, and evaluation."""
    final_solution: str
    judge_mode: JudgeMode
    judge_reasoning: str
    selected_debater_id: str | None
    all_solutions: list[FinalDebaterResponse]
    evaluation_result: EvaluationResult
    success: bool


# =============================================================================
# Abstract Base Classes
# =============================================================================


class DomainConfig(ABC):
    """Abstract configuration for a problem domain."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        pass

    @abstractmethod
    def get_initial_prompt_template(self) -> str:
        pass

    @abstractmethod
    def get_refinement_prompt_template(self) -> str:
        pass

    @abstractmethod
    def extract_solution(self, response: str) -> str:
        pass

    # Total budget (in characters) for ALL neighbor reasoning combined in a
    # single refinement prompt. The orchestration layer (agents.py) checks
    # the actual cumulative length per round and only triggers truncation
    # when this budget is exceeded — by default neighbor reasoning passes
    # through in full. ~8000 chars ≈ 2k tokens, which is small relative to
    # any modern context window. Domains can raise or lower this.
    SHARED_EXPLANATION_BUDGET_CHARS = 8000

    def format_solution_for_sharing(
        self,
        solution: str,
        explanation: str,
        debater_id: str,
        *,
        max_chars: Optional[int] = None,
    ) -> str:
        """Format another debater's response for inclusion in the round-to-round
        refinement prompt.

        Per the canonical multi-agent-debate design pattern (Du et al. 2023;
        AutoGen MAD docs), neighbors must see each other's *full reasoning*
        between rounds — not just the extracted final answer — otherwise
        debaters cannot meaningfully refine their position.

        ``max_chars`` is None by default, meaning the full explanation is
        included verbatim. The call site in ``agents.py`` decides when to
        impose a per-neighbor cap based on the cumulative budget, so
        truncation only fires when the total neighbor reasoning would
        exceed ``SHARED_EXPLANATION_BUDGET_CHARS``.
        """
        reasoning = (explanation or "").strip()
        if max_chars is not None and len(reasoning) > max_chars:
            reasoning = reasoning[:max_chars] + "  ...[truncated]"
        if reasoning:
            return (
                f"=== Response from {debater_id} ===\n"
                f"Reasoning:\n{reasoning}\n\n"
                f"Their answer: {solution}\n"
            )
        return f"=== Response from {debater_id} ===\nTheir answer: {solution}\n"


class Judge(ABC):
    """Abstract judge for aggregating debater solutions."""

    @property
    @abstractmethod
    def mode(self) -> JudgeMode:
        pass

    @abstractmethod
    async def judge(
        self,
        solutions: list[FinalDebaterResponse],
        task: DebateTask,
    ) -> JudgeResponse:
        pass


class SolutionEvaluator(ABC):
    """Abstract evaluator for testing solutions."""

    @abstractmethod
    async def evaluate(self, solution: str, task: DebateTask) -> EvaluationResult:
        pass
