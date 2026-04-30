"""
Shared scenario-side logging helpers.

Scenarios call ``emit_run_summary(messages, rounds=...)`` near the end of
``main()`` to print canonical summary lines that ``run_cmd`` later parses
back into ``result.json``:

    [TOKENS_TOTAL] prompt_tokens=<int> completion_tokens=<int> reasoning_tokens=<int>
    [ROUNDS_TOTAL] rounds=<int>          # only when the scenario passes a round count

Each scenario decides what a "round" means for its own topology — the
framework does not try to infer one. ``rounds=None`` (or omitted) means
"not applicable / not measured" and is recorded as null in result.json.

Per-LLM-call reporting is unified across scenarios via ``emit_llm_call``,
which prints a canonical line and stashes a record for the end-of-run
per-agent summary table. Scenarios pass an "agent" identifier (whatever
that means for the topology — e.g. coder/web_surfer/Assistant/DebaterA/
judge/orchestrator) plus optional extras (label, role, round_idx, etc.):

    [LLM_CALL] agent=<id> duration_s=<f|None> prompt_tokens=<n|None> \\
               completion_tokens=<n|None> reasoning_tokens=<n|None> [extras...]

At end of run, ``emit_run_summary`` automatically prints a per-agent
breakdown table from these records — the totals can't be aggregated into
result.json because each scenario has different agent identifiers, but
the human-readable table makes per-agent cost trivially auditable in
``console_log.txt``.
"""

from __future__ import annotations

import sys
import threading
from typing import Any, Iterable, List, MutableMapping, Optional


# ---------------------------------------------------------------------------
# Token aggregation helpers (used internally by emit_run_summary).
# ---------------------------------------------------------------------------


def _coerce_int(value: object) -> int:
    """Best-effort int coercion. Returns 0 for None / falsy / non-numeric."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def sum_token_usage(messages_or_usages: Iterable[object]) -> tuple[int, int, int]:
    """
    Sum prompt / completion / reasoning tokens across an iterable.

    Accepts either a sequence of agentchat messages (``msg.models_usage``
    is read; messages with ``models_usage is None`` are skipped) or a
    sequence of ``RequestUsage``-like objects with ``prompt_tokens`` /
    ``completion_tokens`` / ``reasoning_tokens`` attributes.
    """
    prompt_total = 0
    completion_total = 0
    reasoning_total = 0
    for item in messages_or_usages:
        # Prefer ``models_usage`` if it exists (agentchat message); else
        # treat the item itself as a usage object.
        usage = getattr(item, "models_usage", item)
        if usage is None:
            continue
        prompt_total += _coerce_int(getattr(usage, "prompt_tokens", 0))
        completion_total += _coerce_int(getattr(usage, "completion_tokens", 0))
        reasoning_total += _coerce_int(getattr(usage, "reasoning_tokens", 0))
    return prompt_total, completion_total, reasoning_total


# ---------------------------------------------------------------------------
# Per-call records (module-level; one process == one scenario).
# ---------------------------------------------------------------------------

_records_lock = threading.Lock()
_call_records: List[MutableMapping[str, Any]] = []


def _format_value(v: Any) -> str:
    """Format a field value for the canonical line. None stays as ``None``."""
    if v is None:
        return "None"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def emit_llm_call(
    *,
    agent: str,
    duration_s: Optional[float] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
    file=None,
    **extras: Any,
) -> None:
    """
    Emit one canonical per-LLM-call line and stash a record for the
    end-of-run per-agent summary.

    The line format is:

        [LLM_CALL] agent=<id> duration_s=<f|None> prompt_tokens=<n|None> \\
                   completion_tokens=<n|None> reasoning_tokens=<n|None> \\
                   [extra_key=value ...]

    ``agent`` is the only required field. Pass ``None`` for any unknown
    measurement (e.g. when ``result.usage`` was not populated by the
    server) — the field is preserved as ``None`` rather than dropped, so
    the line shape is uniform across scenarios.

    Extra fields (e.g. ``label="finalizer"``, ``role="orchestrator"``,
    ``round_idx=2``, ``debater_id="DebaterA"``) are appended verbatim and
    also stashed in the record for the summary.
    """
    if file is None:
        file = sys.stdout

    parts = [
        "[LLM_CALL]",
        f"agent={agent}",
        f"duration_s={_format_value(duration_s)}",
        f"prompt_tokens={_format_value(prompt_tokens)}",
        f"completion_tokens={_format_value(completion_tokens)}",
        f"reasoning_tokens={_format_value(reasoning_tokens)}",
    ]
    for k, v in extras.items():
        parts.append(f"{k}={_format_value(v)}")
    print(" ".join(parts), file=file, flush=True)

    record: MutableMapping[str, Any] = {
        "agent": agent,
        "duration_s": duration_s,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    record.update(extras)
    with _records_lock:
        _call_records.append(record)


def reset_llm_call_records() -> None:
    """Clear the in-memory per-call record buffer. Mainly for tests."""
    with _records_lock:
        _call_records.clear()


def get_llm_call_records() -> List[MutableMapping[str, Any]]:
    """Return a copy of the per-call records collected so far."""
    with _records_lock:
        return [dict(r) for r in _call_records]


# ---------------------------------------------------------------------------
# Per-agent summary table (printed to console_log.txt at end of run).
# ---------------------------------------------------------------------------


def _format_summary_table(records: List[MutableMapping[str, Any]]) -> str:
    """Build a fixed-width per-agent summary table from call records."""
    if not records:
        return ""

    # Aggregate per agent. Order = first-seen order, so the table reads
    # naturally for the scenario's topology rather than alphabetically.
    seen: List[str] = []
    by_agent: dict[str, dict[str, Any]] = {}
    for r in records:
        agent = str(r.get("agent", "?"))
        if agent not in by_agent:
            seen.append(agent)
            by_agent[agent] = {
                "calls": 0,
                "prompt": 0,
                "completion": 0,
                "reasoning": 0,
                "duration": 0.0,
                # Track when measurements are missing so we can flag them.
                "missing_tokens": 0,
                "missing_duration": 0,
            }
        a = by_agent[agent]
        a["calls"] += 1
        if r.get("prompt_tokens") is None and r.get("completion_tokens") is None:
            a["missing_tokens"] += 1
        a["prompt"] += _coerce_int(r.get("prompt_tokens"))
        a["completion"] += _coerce_int(r.get("completion_tokens"))
        a["reasoning"] += _coerce_int(r.get("reasoning_tokens"))
        d = r.get("duration_s")
        if d is None:
            a["missing_duration"] += 1
        else:
            try:
                a["duration"] += float(d)
            except (TypeError, ValueError):
                a["missing_duration"] += 1

    # Column widths chosen for compactness; agent column expands to fit.
    agent_w = max(len("agent"), max(len(s) for s in seen))
    headers = (
        "agent".ljust(agent_w),
        "calls".rjust(6),
        "prompt".rjust(11),
        "completion".rjust(11),
        "reasoning".rjust(11),
        "duration_s".rjust(11),
    )
    sep = ("-" * agent_w, "-" * 6, "-" * 11, "-" * 11, "-" * 11, "-" * 11)

    lines = ["=== Per-agent LLM call summary ==="]
    lines.append("  " + "  ".join(headers))
    lines.append("  " + "  ".join(sep))

    total_calls = total_p = total_c = total_r = 0
    total_d = 0.0
    for agent in seen:
        a = by_agent[agent]
        flag = ""
        if a["missing_tokens"] or a["missing_duration"]:
            flag = (
                f"  ({a['missing_tokens']} call(s) missing tokens, "
                f"{a['missing_duration']} missing duration)"
                if (a["missing_tokens"] and a["missing_duration"])
                else (
                    f"  ({a['missing_tokens']} call(s) missing tokens)"
                    if a["missing_tokens"]
                    else f"  ({a['missing_duration']} call(s) missing duration)"
                )
            )
        lines.append(
            "  "
            + "  ".join(
                (
                    agent.ljust(agent_w),
                    str(a["calls"]).rjust(6),
                    str(a["prompt"]).rjust(11),
                    str(a["completion"]).rjust(11),
                    str(a["reasoning"]).rjust(11),
                    f"{a['duration']:.3f}".rjust(11),
                )
            )
            + flag
        )
        total_calls += a["calls"]
        total_p += a["prompt"]
        total_c += a["completion"]
        total_r += a["reasoning"]
        total_d += a["duration"]

    lines.append("  " + "  ".join(sep))
    lines.append(
        "  "
        + "  ".join(
            (
                "TOTAL".ljust(agent_w),
                str(total_calls).rjust(6),
                str(total_p).rjust(11),
                str(total_c).rjust(11),
                str(total_r).rjust(11),
                f"{total_d:.3f}".rjust(11),
            )
        )
    )
    return "\n".join(lines)


def emit_per_agent_summary(file=None) -> None:
    """
    Print a fixed-width per-agent breakdown of every ``[LLM_CALL]`` line
    emitted so far. No-op if no calls have been recorded.

    Called automatically by ``emit_run_summary``; scenarios can also call
    it directly if they want to flush the table at a different point.
    """
    if file is None:
        file = sys.stdout
    with _records_lock:
        records = list(_call_records)
    table = _format_summary_table(records)
    if table:
        print(table, file=file, flush=True)


# ---------------------------------------------------------------------------
# End-of-run summary (parsed back into result.json by run_cmd).
# ---------------------------------------------------------------------------


def emit_run_summary(
    messages_or_usages: Iterable[object],
    *,
    rounds: Optional[int] = None,
    file=None,
) -> None:
    """
    Emit canonical end-of-run summary lines parsed by ``run_cmd`` into
    ``result.json``, and a human-readable per-agent breakdown table.

    Args:
        messages_or_usages: iterable of agentchat messages or usage objects.
        rounds: scenario-defined round count. ``None`` if not applicable.
        file: optional stream override (defaults to ``sys.stdout``).
    """
    if file is None:
        file = sys.stdout

    p, c, r = sum_token_usage(messages_or_usages)
    print(
        f"[TOKENS_TOTAL] prompt_tokens={p} completion_tokens={c} reasoning_tokens={r}",
        file=file,
        flush=True,
    )
    if rounds is not None:
        print(f"[ROUNDS_TOTAL] rounds={int(rounds)}", file=file, flush=True)

    # Print the per-agent breakdown if any [LLM_CALL] lines were emitted.
    emit_per_agent_summary(file=file)
