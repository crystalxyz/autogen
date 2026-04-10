"""LMEval ReAct-style agent scenario runner.

Wraps lm-evaluation-harness with a single-agent ReAct loop:

  - One `AssistantAgent` ("solver") with native tool calling, no team,
    no group chat, no code-block-string parsing.
  - Five default tools, all offline / deterministic / no API keys:
      python_exec       — sandboxed Python subprocess
      calculator        — sympy expression evaluation
      physical_constant — scipy.constants lookup
      unit_convert      — pint unit conversion
      final_answer      — solver calls this to submit and terminate
  - The agent itself runs the ReAct loop internally via
    `max_tool_iterations` + `reflect_on_tool_use=True` so a single
    `on_messages()` call drives the whole chain (Thought → Tool →
    Observation → Thought → ... → final_answer call → done), bounded
    by max_turns.
  - Termination: the `final_answer` tool stores the answer in a
    per-question collector. After on_messages returns we read the
    collector. Fallback to text extraction if the model emitted a
    text answer without using the tool.

Backend: an autogen `ChatCompletionClient` built from `config.yaml`,
identical to HumanEval. The base_url is whatever `run_bench.py` (or the
direct `agbench run -c`) injected.

Per-request metrics captured: model calls, tool calls (total + per
tool), trace messages, terminated_by. Per-call latencies via a
monkey-patch on the client. Full trace + answer + metadata is embedded
in lm_eval_result.json under `agent_traces[task][i]`.
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Force line-buffered stdout/stderr so every print() (including the
# per-request `[lmeval_agent_step]` markers) flushes to console_log.txt
# the moment it is emitted, not in 4-8 KB chunks. Critical for long
# scenarios where the process can be killed by an external timeout
# before block-buffered prints would otherwise reach disk. Independent
# of and complementary to PYTHONUNBUFTERED / `python -u`.
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except AttributeError:
    # Python <3.7 fallback (shouldn't apply, agbench requires 3.10+).
    pass

import yaml

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
)
from autogen_core import CancellationToken
from autogen_core.model_context import UnboundedChatCompletionContext
from autogen_core.models import ChatCompletionClient

from lm_eval.api.model import LM
from lm_eval import simple_evaluate


# =============================================================================
# Task-specific answer formatters (lm-eval task filters expect specific
# canonical formats; the ReAct agent yields a clean answer string and we wrap
# it appropriately for each task before returning to lm-eval).
# =============================================================================
def _format_gpqa(answer: str) -> str:
    a = answer.strip()
    m = re.search(r"\(?\s*([A-Da-d])\s*\)?", a)
    letter = m.group(1).upper() if m else a.strip("().,").upper()
    return f"The answer is ({letter})."


TASK_ANSWER_FORMATTERS = {
    "gsm8k": lambda a: f"#### {a}",
    "gsm8k_cot": lambda a: f"#### {a}",
    "gpqa_diamond_cot_zeroshot": _format_gpqa,
    "gpqa_main_cot_zeroshot": _format_gpqa,
    "gpqa_extended_cot_zeroshot": _format_gpqa,
}


def _format_answer_for_task(task_name: str, answer: str) -> str:
    fmt = TASK_ANSWER_FORMATTERS.get(task_name)
    return fmt(answer) if fmt else answer


# =============================================================================
# Per-request latency capture: monkey-patch model_client.create() so every
# autogen LLM call is timed and recorded into _LATENCIES.
# =============================================================================
_LATENCIES: List[dict] = []


def _install_client_latency_hook(client: ChatCompletionClient) -> None:
    orig_create = client.create

    async def timed_create(*args, **kwargs):
        t0 = time.perf_counter()
        status = "ok"
        err: Optional[str] = None
        try:
            return await orig_create(*args, **kwargs)
        except Exception as exc:
            status = "error"
            err = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            t1 = time.perf_counter()
            _LATENCIES.append({
                "index": len(_LATENCIES),
                "start": t0,
                "end": t1,
                "elapsed": t1 - t0,
                "status": status,
                "error": err,
            })

    client.create = timed_create  # type: ignore[assignment]


def _latency_summary(latencies: List[dict]) -> dict:
    if not latencies:
        return {}
    ok = [e["elapsed"] for e in latencies if e["status"] == "ok"]
    if not ok:
        return {"n": 0, "n_error": len(latencies)}
    ok_sorted = sorted(ok)
    n = len(ok_sorted)

    def pct(q: float) -> float:
        return ok_sorted[min(n - 1, int(q * n))]

    return {
        "n": n,
        "n_error": len(latencies) - n,
        "mean": sum(ok_sorted) / n,
        "min": ok_sorted[0],
        "max": ok_sorted[-1],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "total": sum(ok_sorted),
    }


# =============================================================================
# Tools
# =============================================================================
PYTHON_EXEC_TIMEOUT_S = 30
PYTHON_EXEC_OUTPUT_LIMIT = 2000


def python_exec(code: str) -> str:
    """Execute Python code in a fresh subprocess and return the combined output.

    Use this for any computation, simulation, list manipulation, or string
    parsing that's faster to run than to reason about. Variables do not
    persist across calls — each invocation runs in a new process. The script
    has 30 seconds to finish; output is truncated to 2000 characters per
    stream.

    Args:
        code: Python source code to execute. Print results you want to see.

    Returns:
        A string containing the exit code, stdout, and stderr.
    """
    script_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(code)
            script_path = fh.name
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=PYTHON_EXEC_TIMEOUT_S,
        )
        out = (result.stdout or "")[:PYTHON_EXEC_OUTPUT_LIMIT]
        err = (result.stderr or "")[:PYTHON_EXEC_OUTPUT_LIMIT]
        parts = [f"exit_code={result.returncode}"]
        if out:
            parts.append(f"--- stdout ---\n{out}")
        if err:
            parts.append(f"--- stderr ---\n{err}")
        return "\n".join(parts) if (out or err) else f"exit_code={result.returncode}\n(no output)"
    except subprocess.TimeoutExpired:
        return f"ERROR: timeout after {PYTHON_EXEC_TIMEOUT_S}s"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"
    finally:
        if script_path:
            try:
                os.unlink(script_path)
            except OSError:
                pass


def calculator(expression: str) -> str:
    """Evaluate a mathematical expression using sympy.

    Supports arbitrary precision arithmetic, algebra, calculus, and equation
    solving. Cheaper than python_exec for one-off calculations. Examples of
    valid expressions:
        '17 * 23'
        'sqrt(2) + pi'
        'integrate(x**2, (x, 0, 1))'
        'solve(x**2 - 4, x)'
        'diff(sin(x), x)'

    Args:
        expression: A sympy-compatible expression as a string.

    Returns:
        The simplified result, plus a numeric value if it evaluates to one.
    """
    try:
        import sympy

        result = sympy.sympify(expression)
        simplified = sympy.simplify(result)
        if hasattr(simplified, "is_number") and simplified.is_number:
            try:
                return f"{simplified} (numeric: {float(simplified)})"
            except (TypeError, ValueError):
                return f"{simplified}"
        return f"{simplified}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def physical_constant(name: str) -> str:
    """Look up a physical constant by name from scipy.constants.

    Performs a case-insensitive substring match against the
    `physical_constants` table. Examples of valid names:
        'Planck constant'
        'speed of light in vacuum'
        'Boltzmann constant'
        'electron mass'
        'elementary charge'
        'Avogadro constant'

    Args:
        name: Constant name (or substring of one).

    Returns:
        Matched constant key, value, unit, and standard uncertainty.
    """
    try:
        from scipy.constants import physical_constants

        nl = name.lower().strip()
        matches = [k for k in physical_constants if nl in k.lower()]
        if not matches:
            return (
                f"ERROR: no constant matching '{name}'. "
                "Try names like 'Planck constant', 'speed of light in vacuum', "
                "'Boltzmann constant', 'electron mass'."
            )
        # Prefer exact name match if present, else first substring match.
        key = next((k for k in matches if k.lower() == nl), matches[0])
        value, unit, uncertainty = physical_constants[key]
        return f"{key}: value={value}, unit='{unit}', uncertainty={uncertainty}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numerical value between units using pint.

    Supports SI, imperial, atomic, astronomical, and CGS units. Useful when
    a question's answer choices are in different units than the units you'd
    naturally compute in. Examples:
        unit_convert(5, 'eV', 'joule')
        unit_convert(1, 'lightyear', 'meter')
        unit_convert(300, 'kelvin', 'celsius')
        unit_convert(1, 'atmosphere', 'pascal')

    Args:
        value: Numerical value to convert.
        from_unit: Source unit string.
        to_unit: Target unit string.

    Returns:
        The converted value with its unit, or an error message.
    """
    try:
        import pint

        ureg = pint.UnitRegistry()
        q = value * ureg(from_unit)
        result = q.to(to_unit)
        return f"{result.magnitude} {result.units}"
    except Exception as exc:
        return f"ERROR: {type(exc).__name__}: {exc}"


class _AnswerCollector:
    """Holds the final answer for a single question. The agent's
    `final_answer` tool sets this idempotently — only the first call
    wins, subsequent calls are no-ops. autogen-agentchat does not
    expose a clean way to terminate AssistantAgent.on_messages from
    inside a tool (CancellationToken from a sync tool deadlocks the
    tool execution loop in 0.7.x), so we rely on max_tool_iterations
    to bound the loop and read the answer back after on_messages
    returns."""

    def __init__(self) -> None:
        self.answer: Optional[str] = None
        self.submitted: bool = False

    def make_tool(self):
        def final_answer(answer: str) -> str:
            """Submit your final answer to the question. Call this exactly
            once, when you have determined the answer.

            The answer should be the answer value only:
              - Multiple choice: just the letter, e.g. 'A' or 'C'.
              - Numerical: just the number, e.g. '42' or '3.14'.
              - Free-form: just the answer text with no preamble.

            Do not include explanations, units (unless asked), or surrounding
            prose. After calling this tool, do not call any more tools — you
            are done.

            Args:
                answer: The final answer value as a plain string.

            Returns:
                Confirmation that the answer was recorded.
            """
            if not self.submitted:
                self.answer = answer
                self.submitted = True
            return (
                f"Answer recorded: {self.answer!r}. "
                "You are done. Do not call any more tools."
            )

        return final_answer


# =============================================================================
# System prompt for the ReAct solver.
# =============================================================================
REACT_SYSTEM_MESSAGE = """You are an expert problem solver. You answer questions \
using step-by-step reasoning and a small set of tools when they would make your \
answer more reliable than reasoning alone.

Available tools (the runtime exposes them via function calling — you do not need \
to write code blocks):
  - python_exec(code): run a Python script in a fresh subprocess; use for \
    iterative computation, list/string manipulation, or anything you'd rather \
    verify than guess.
  - calculator(expression): evaluate a sympy expression for one-off arithmetic, \
    algebra, calculus, or equation solving.
  - physical_constant(name): look up a physical constant (Planck, speed of \
    light, Boltzmann, etc.) with its exact value, unit, and uncertainty.
  - unit_convert(value, from_unit, to_unit): convert between units (eV ↔ J, \
    K ↔ °C, lightyear ↔ meter, ...).
  - final_answer(answer): submit your final answer.

Guidelines:
  1. Think briefly about what the question is really asking.
  2. Use tools when they would help — don't reach for python_exec if a single \
     calculator call suffices, and don't call any tool for a simple lookup you \
     already know.
  3. When you have determined the answer, call `final_answer` with the answer \
     value only:
       - Multiple choice → just the letter (A, B, C, or D). Pass the LETTER, \
         not the option's value. For example, if option (D) is "-0.7" and your \
         computed answer is -0.7, you must call final_answer("D"), NOT \
         final_answer("-0.7").
       - Numerical (open-ended) → just the number, no units unless asked.
       - Free-form → just the answer text, no preamble.
     Do NOT put the final answer in a plain text reply — you MUST call the \
     `final_answer` tool to submit it.
"""


# =============================================================================
# Trace conversion / metric extraction
# =============================================================================
def _message_to_trace(msg: Any, idx: int) -> dict:
    """Convert any agentchat message into a JSON-serializable trace entry."""
    entry: dict = {
        "msg_index": idx,
        "type": type(msg).__name__,
        "source": getattr(msg, "source", None),
    }
    content = getattr(msg, "content", None)

    if isinstance(msg, ToolCallRequestEvent):
        entry["tool_calls"] = [
            {"name": getattr(c, "name", None), "arguments": getattr(c, "arguments", None)}
            for c in (content or [])
        ]
    elif isinstance(msg, ToolCallExecutionEvent):
        entry["tool_results"] = [
            {
                "name": getattr(r, "name", None),
                "is_error": getattr(r, "is_error", False),
                "content": str(getattr(r, "content", ""))[:1500],
            }
            for r in (content or [])
        ]
    elif isinstance(content, str):
        entry["content"] = content
    elif content is None:
        entry["content"] = None
    else:
        try:
            entry["content"] = str(content)[:1500]
        except Exception:
            entry["content"] = ""

    return entry


def _summarize_trace(trace: List[dict], max_turns: int) -> dict:
    """Compute structured per-request metrics from a flat trace."""
    num_user_msgs = 0
    num_solver_msgs = 0
    num_tool_calls = 0
    num_tool_errors = 0
    tool_call_counts: Dict[str, int] = {}

    for m in trace:
        ty = m.get("type", "")
        src = m.get("source")
        if src == "user":
            num_user_msgs += 1
        elif src == "solver" and ty in ("TextMessage", "ToolCallSummaryMessage"):
            num_solver_msgs += 1

        if ty == "ToolCallRequestEvent":
            for tc in m.get("tool_calls", []):
                num_tool_calls += 1
                name = tc.get("name") or "<unknown>"
                tool_call_counts[name] = tool_call_counts.get(name, 0) + 1
        elif ty == "ToolCallExecutionEvent":
            for tr in m.get("tool_results", []):
                if tr.get("is_error"):
                    num_tool_errors += 1

    return {
        "num_trace_messages": len(trace),
        "num_user_msgs": num_user_msgs,
        "num_solver_msgs": num_solver_msgs,
        "num_tool_calls": num_tool_calls,
        "num_tool_errors": num_tool_errors,
        "tool_call_counts": tool_call_counts,
    }


# =============================================================================
# The lm-eval LM subclass
# =============================================================================
class AutogenReActLM(LM):
    """lm-eval LM that routes generation through an autogen AssistantAgent
    running a ReAct loop with native tool calling."""

    def __init__(
        self,
        model_client: ChatCompletionClient,
        max_turns: int = 4,
        task_name: str = "",
    ) -> None:
        super().__init__()
        self._model_client = model_client
        self._max_turns = max_turns
        self._task_name = task_name
        self._rank = 0
        self._world_size = 1
        # Parallel to lm-eval's samples list: one entry per generate_until
        # request, in arrival order.
        self.traces: List[dict] = []

    # -- required abstract methods ------------------------------------------
    def loglikelihood(self, requests):  # type: ignore[override]
        raise NotImplementedError(
            "AutogenReActLM only supports generation tasks. "
            "Use a generate_until task like gsm8k, gpqa_*_cot_zeroshot, triviaqa."
        )

    def loglikelihood_rolling(self, requests):  # type: ignore[override]
        raise NotImplementedError("AutogenReActLM only supports generation tasks.")

    def generate_until(self, requests) -> List[str]:  # type: ignore[override]
        results: List[str] = []
        for i, instance in enumerate(requests):
            context, _gen_kwargs = instance.args
            print(f"\n========== REQUEST {i} ==========")
            calls_before = len(_LATENCIES)
            try:
                raw_answer, trace, terminated_by = asyncio.run(self._solve(context))
                # MCQ recovery: if this is a multiple-choice task and the
                # model returned a non-letter (e.g. submitted the option
                # value instead of its letter), parse the option list out
                # of the question and look the value up.
                if (
                    self._task_name in TASK_ANSWER_FORMATTERS
                    and self._task_name.startswith("gpqa")
                    and raw_answer
                    and not _LETTER_FULLMATCH_RE.fullmatch(raw_answer.strip())
                ):
                    recovered_letter = _recover_mcq_letter(context, raw_answer)
                    if recovered_letter:
                        print(
                            f"[lmeval_mcq_recovery] request={i} "
                            f"raw={raw_answer!r} -> letter={recovered_letter!r}"
                        )
                        raw_answer = recovered_letter
                        terminated_by = terminated_by + "+mcq_recovered"
                summary = _summarize_trace(trace, self._max_turns)
                model_calls = len(_LATENCIES) - calls_before
                self.traces.append({
                    "request_index": i,
                    "status": "ok",
                    "raw_answer": raw_answer,
                    "formatted_answer": _format_answer_for_task(self._task_name, raw_answer),
                    "num_model_calls": model_calls,
                    "terminated_by": terminated_by,
                    **summary,
                    "trace": trace,
                })
            except Exception as exc:
                print(f"[lmeval_agent_error] request={i} raised:")
                traceback.print_exc()
                raw_answer = ""
                self.traces.append({
                    "request_index": i,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_answer": "",
                    "formatted_answer": "",
                    "num_model_calls": len(_LATENCIES) - calls_before,
                    "terminated_by": "error",
                    "num_trace_messages": 0,
                    "num_user_msgs": 0,
                    "num_solver_msgs": 0,
                    "num_tool_calls": 0,
                    "num_tool_errors": 0,
                    "tool_call_counts": {},
                    "trace": [],
                })

            formatted = _format_answer_for_task(self._task_name, raw_answer)
            results.append(formatted)
            last = self.traces[-1]
            tcc = last.get("tool_call_counts", {})
            print(
                f"[lmeval_agent_step] request={i} "
                f"raw={raw_answer!r} formatted={formatted!r} "
                f"model_calls={last['num_model_calls']} "
                f"tool_calls={last.get('num_tool_calls', 0)} "
                f"tool_errors={last.get('num_tool_errors', 0)} "
                f"tools={tcc} "
                f"terminated_by={last['terminated_by']}"
            )
        return results

    async def _solve(self, question: str):
        """Run one ReAct loop for a single question.

        Returns (raw_answer, trace, terminated_by).
        """
        collector = _AnswerCollector()

        # Build a fresh agent per question so chat context, tool state, and
        # latency attribution are clean.
        solver = AssistantAgent(
            name="solver",
            model_client=self._model_client,
            tools=[
                python_exec,
                calculator,
                physical_constant,
                unit_convert,
                collector.make_tool(),
            ],
            system_message=REACT_SYSTEM_MESSAGE,
            model_context=UnboundedChatCompletionContext(),
            reflect_on_tool_use=True,
            # Bound the internal Thought→Tool→Observation loop. Each
            # iteration is one model call possibly followed by tool
            # invocation. After max_tool_iterations, the agent forces a
            # final reply even if the model wanted to keep using tools.
            # We don't have a way to terminate early when final_answer is
            # called (CancellationToken from a sync tool deadlocks
            # autogen 0.7.x), so set this fairly low — the model will
            # likely call final_answer in 1-3 iterations and waste a few
            # extra calls afterward, but the collector is idempotent.
            max_tool_iterations=max(2, self._max_turns),
        )

        user_msg = TextMessage(content=question, source="user")
        response = await solver.on_messages([user_msg], CancellationToken())

        # Build the flat trace: user message, then every inner_message
        # (Thought / ToolCallRequest / ToolCallExecution / ...) in order,
        # then the final chat_message.
        trace: List[dict] = [_message_to_trace(user_msg, 0)]
        idx = 1
        for m in response.inner_messages or []:
            trace.append(_message_to_trace(m, idx))
            idx += 1
        if response.chat_message is not None:
            trace.append(_message_to_trace(response.chat_message, idx))

        # Determine the answer + termination reason in priority order:
        #   1. final_answer tool was called → use the recorded answer.
        #   2. Hit max_tool_iterations → fallback to final chat_message text.
        #   3. Otherwise → fallback to final chat_message text.
        if collector.submitted and collector.answer is not None:
            return collector.answer, trace, "final_answer_tool"

        # Pull text from the final chat_message for the fallback paths.
        final_text = ""
        if response.chat_message is not None:
            content = getattr(response.chat_message, "content", "")
            if isinstance(content, str):
                final_text = content

        # Recover when the server didn't parse tool calls into the
        # OpenAI tool_calls field — the model's <tool_call> JSON
        # arrives as raw text in the chat message.
        recovered = _recover_final_answer_from_text(final_text)
        if recovered is not None:
            return recovered, trace, "text_recovered_tool_call"

        # Legacy "FINAL_ANSWER:" plain-text marker fallback.
        m = re.search(r"FINAL_ANSWER:\s*(.+)", final_text)
        if m:
            return m.group(1).strip().splitlines()[0].strip(), trace, "fallback_text_marker"
        return final_text.strip(), trace, "fallback_no_tool_call"


# =============================================================================
# Substitution + main
# =============================================================================
# =============================================================================
# Defensive recovery: when the SGLang server isn't running with the
# right --tool-call-parser, Qwen3's `<tool_call>...</tool_call>` JSON
# blocks come through as raw text in the chat message instead of being
# parsed into structured tool_calls. Scan the text for an embedded
# final_answer invocation and recover it.
# =============================================================================
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FINAL_ANSWER_PY_RE = re.compile(r"final_answer\s*\(\s*['\"]?([^'\"\)]+?)['\"]?\s*\)")
_MCQ_OPTION_RE = re.compile(r"\(([A-D])\)\s*(.+?)(?=\n\([A-D]\)|\Z)", re.DOTALL)
_LETTER_FULLMATCH_RE = re.compile(r"\(?\s*([A-Da-d])\s*\)?\.?")


def _recover_mcq_letter(question: str, value_answer: str) -> Optional[str]:
    """For multiple-choice tasks, when the model submits the value of an
    option (e.g. '-0.7') instead of its letter ('D'), parse the
    "(A) val\n(B) val\n(C) val\n(D) val" block out of the question and
    look the value up. Returns the recovered letter, or None."""
    if not question or not value_answer:
        return None
    options: Dict[str, str] = {}
    for m in _MCQ_OPTION_RE.finditer(question):
        options[m.group(1)] = m.group(2).strip().rstrip(".,")
    if not options:
        return None
    target = value_answer.strip().rstrip(".,").strip()
    # 1. exact string match
    for letter, value in options.items():
        if value == target or value.lstrip("+") == target.lstrip("+"):
            return letter
    # 2. numeric match (handles 0.7 vs 0.70 vs +0.7 etc.)
    try:
        target_num = float(target)
        for letter, value in options.items():
            try:
                if float(value) == target_num:
                    return letter
            except (TypeError, ValueError):
                continue
    except (TypeError, ValueError):
        pass
    # 3. substring containment (e.g. answer="boson", option="A boson particle")
    tlower = target.lower()
    for letter, value in options.items():
        if tlower and tlower in value.lower():
            return letter
    return None


def _recover_final_answer_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    # 1. Qwen-style structured tool call: <tool_call>{...}</tool_call>
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name") == "final_answer":
            args = obj.get("arguments") or {}
            ans = args.get("answer")
            if ans is not None:
                return str(ans)
    # 2. Bare Python call syntax: final_answer("X") or final_answer(X)
    m = _FINAL_ANSWER_PY_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def read_placeholder(path: str, default: str) -> str:
    with open(path, "rt") as fh:
        value = fh.read().strip()
    if value.startswith("__") and value.endswith("__"):
        return default
    return value


def main() -> None:
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(cfg["model_config"])
    _install_client_latency_hook(model_client)

    mc = cfg["model_config"]["config"]
    print(
        f"[lmeval_agent_config] model={mc.get('model')} "
        f"base_url={mc.get('base_url')}"
    )

    task_name = read_placeholder("task_name.txt", "gsm8k")
    num_fewshot = int(read_placeholder("fewshot.txt", "0") or "0")
    limit_str = read_placeholder("limit.txt", "0") or "0"
    max_turns = int(read_placeholder("max_turns.txt", "4") or "4")
    limit_val: Any = int(limit_str)
    if limit_val <= 0:
        limit_val = None

    print(
        f"[lmeval_agent_task] task={task_name} num_fewshot={num_fewshot} "
        f"limit={limit_val} max_turns={max_turns}"
    )

    lm = AutogenReActLM(
        model_client=model_client,
        max_turns=max_turns,
        task_name=task_name,
    )

    start = time.time()
    success = False
    results: dict = {}
    try:
        results = simple_evaluate(
            model=lm,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            limit=limit_val,
            batch_size=1,
            log_samples=True,
            apply_chat_template=False,  # we drive prompting via the system message
        ) or {}

        task_results = results.get("results", {}).get(task_name)
        if task_results is None:
            print(f"[lmeval_agent_error] no results for task={task_name}")
        else:
            for metric, value in task_results.items():
                if metric == "alias" or metric.endswith("_stderr,none"):
                    continue
                print(f"[lmeval_metric] task={task_name} metric={metric} value={value}")
            success = True
    except Exception:
        print("[lmeval_agent_error] simple_evaluate raised:")
        traceback.print_exc()

    elapsed = time.time() - start

    if isinstance(results, dict):
        results["agent_traces"] = {task_name: lm.traces}
        results["latencies"] = _LATENCIES
        results["latency_stats"] = _latency_summary(_LATENCIES)
        if lm.traces:
            def _dist(key: str) -> dict:
                xs = [t.get(key, 0) for t in lm.traces]
                if not xs:
                    return {}
                return {
                    "mean": sum(xs) / len(xs),
                    "min": min(xs),
                    "max": max(xs),
                    "total": sum(xs),
                }

            term_counts: Dict[str, int] = {}
            for t in lm.traces:
                term_counts[t.get("terminated_by", "unknown")] = (
                    term_counts.get(t.get("terminated_by", "unknown"), 0) + 1
                )
            tool_use_total: Dict[str, int] = {}
            for t in lm.traces:
                for k, v in (t.get("tool_call_counts") or {}).items():
                    tool_use_total[k] = tool_use_total.get(k, 0) + v

            results["agent_stats"] = {
                "requests": len(lm.traces),
                "model_calls": _dist("num_model_calls"),
                "solver_msgs": _dist("num_solver_msgs"),
                "tool_calls": _dist("num_tool_calls"),
                "tool_errors": _dist("num_tool_errors"),
                "trace_messages": _dist("num_trace_messages"),
                "terminated_by": term_counts,
                "tool_use_total": tool_use_total,
            }

    stats = results.get("latency_stats") if isinstance(results, dict) else None
    if stats:
        print(
            f"[lmeval_latency] n={stats['n']} mean={stats['mean']:.3f}s "
            f"p50={stats['p50']:.3f}s p90={stats['p90']:.3f}s "
            f"p99={stats['p99']:.3f}s min={stats['min']:.3f}s "
            f"max={stats['max']:.3f}s total={stats['total']:.3f}s"
        )
    ats = results.get("agent_stats") if isinstance(results, dict) else None
    if ats:
        mc_d = ats.get("model_calls", {})
        tc_d = ats.get("tool_calls", {})
        print(
            f"[lmeval_agent_stats] requests={ats['requests']} "
            f"model_calls=(mean={mc_d.get('mean', 0):.2f} "
            f"total={mc_d.get('total', 0)}) "
            f"tool_calls=(mean={tc_d.get('mean', 0):.2f} "
            f"total={tc_d.get('total', 0)}) "
            f"tool_use={ats.get('tool_use_total', {})} "
            f"terminated_by={ats.get('terminated_by', {})}"
        )

    try:
        with open("lm_eval_result.json", "w") as fh:
            json.dump(results, fh, indent=2, default=str)
    except Exception:
        traceback.print_exc()

    print(f"LMEval execution time: {elapsed:.2f} seconds")

    if success:
        print("ALL TESTS PASSED !#!#")
    else:
        print("SOME TESTS FAILED !#!#")

    # Final hard flush before exit, in case anything is still pending.
    sys.stdout.flush()
    sys.stderr.flush()

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
