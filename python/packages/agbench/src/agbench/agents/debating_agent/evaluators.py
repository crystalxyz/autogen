"""
Evaluators for testing solutions in multi-agent debate.
"""

import math
import re

from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor

from .protocol import DebateTask, EvaluationResult, SolutionEvaluator


class PassThroughEvaluator(SolutionEvaluator):
    """Evaluator that always passes (no validation)."""

    async def evaluate(self, solution: str, task: DebateTask) -> EvaluationResult:
        return EvaluationResult(passed=True, score=1.0)


class CodeExecutionEvaluator(SolutionEvaluator):
    """Evaluator that tests code by executing it against test cases."""

    def __init__(
        self,
        test_code: str,
        entry_point: str,
        work_dir: str = ".",
    ) -> None:
        self._test_code = test_code
        self._entry_point = entry_point
        self._executor = LocalCommandLineCodeExecutor(work_dir=work_dir)

    async def evaluate(self, solution: str, task: DebateTask) -> EvaluationResult:
        full_code = f"""{self._test_code}

def run_tests(candidate):
    try:
        check(candidate)
        return True
    except AssertionError:
        return False
    except Exception:
        return False

{solution}

result = run_tests({self._entry_point})
print("TEST_RESULT:", result)
"""
        try:
            result = await self._executor.execute_code_blocks(
                [CodeBlock(code=full_code, language="python")],
                cancellation_token=CancellationToken(),
            )
            passed = "TEST_RESULT: True" in result.output
            return EvaluationResult(
                passed=passed,
                score=1.0 if passed else 0.0,
                details={"output": result.output[:2000]},
                error=None if passed else result.output,
            )
        except Exception as e:
            return EvaluationResult(
                passed=False,
                score=0.0,
                error=f"Execution error: {e}",
            )


class MathEvaluator(SolutionEvaluator):
    """Evaluator that compares numerical answers."""

    def __init__(
        self,
        expected_answer: float | int | str | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        self._expected_answer = expected_answer
        self._tolerance = tolerance

    async def evaluate(self, solution: str, task: DebateTask) -> EvaluationResult:
        expected = self._expected_answer or task.context.get("expected_answer")
        if expected is None:
            return EvaluationResult(passed=False, error="No expected answer provided")

        try:
            expected_num = float(str(expected).replace(",", ""))
        except ValueError:
            return EvaluationResult(passed=False, error=f"Invalid expected answer: {expected}")

        # Extract number from solution
        extracted = self._extract_number(solution)
        if extracted is None:
            return EvaluationResult(
                passed=False,
                error="Could not extract number from solution",
                details={"solution": solution[:200]},
            )

        is_correct = math.isclose(extracted, expected_num, rel_tol=self._tolerance)
        return EvaluationResult(
            passed=is_correct,
            score=1.0 if is_correct else 0.0,
            details={"extracted": extracted, "expected": expected_num},
        )

    def _extract_number(self, text: str) -> float | None:
        # Try {{X}} format
        match = re.search(r'\{\{([^}]+)\}\}', text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

        # Find last number
        numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                pass
        return None


class ExactMatchEvaluator(SolutionEvaluator):
    """Evaluator that requires exact string match."""

    def __init__(
        self,
        expected_answer: str | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self._expected_answer = expected_answer
        self._case_sensitive = case_sensitive

    async def evaluate(self, solution: str, task: DebateTask) -> EvaluationResult:
        expected = self._expected_answer or task.context.get("expected_answer")
        if expected is None:
            return EvaluationResult(passed=False, error="No expected answer provided")

        sol = solution.strip()
        exp = str(expected).strip()

        if not self._case_sensitive:
            sol = sol.lower()
            exp = exp.lower()

        is_match = sol == exp
        return EvaluationResult(
            passed=is_match,
            score=1.0 if is_match else 0.0,
            details={"solution": solution[:200], "expected": str(expected)[:200]},
        )
