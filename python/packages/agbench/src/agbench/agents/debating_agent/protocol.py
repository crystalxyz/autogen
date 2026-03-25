"""
Message protocol and base interfaces for multi-agent debate.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

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

    def format_solution_for_sharing(self, solution: str, debater_id: str) -> str:
        return f"=== Solution from {debater_id} ===\n{solution}\n"


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
