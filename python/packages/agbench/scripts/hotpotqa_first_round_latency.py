"""First-round latency breakdown for HotpotQA SelectorGroupChat runs.

Per rep, isolate ONE round of activity (the very first one) and split its
wall time into three buckets:

    orch/selector  -> sum of [LATENCY] orchestrator selector calls before
                      the first [runtime] event (handles selector retries)
    agent          -> the first [runtime] seconds value (already includes
                      the agent's own LLM call + browser/tool wall time)
    orch/terminator-> sum of [LATENCY] orchestrator termination calls
                      between that [runtime] and the next selector call
    round_e2e      -> sum of the three above

Round 1 has the smallest selector context so it isolates per-call
inference speed at a workload that is comparable across runs. Don't
extrapolate to whole-rep numbers - later rounds carry much larger
selector context. Thinking (-t) variants pay extra in selector and
termination due to reasoning tokens; compare same-mode pairs.

Outputs reports/csv/hotpotqa_first_round_latency.csv plus three console
tables (agent mix, overall first-round breakdown, conditional breakdown
by which agent was picked in round 1).
"""

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path("/home/xz957/autogen/python/packages/agbench")
OUT = ROOT / "outputs"
CSV_OUT = ROOT / "reports" / "csv" / "hotpotqa_first_round_latency.csv"

LATENCY_RE = re.compile(
    r"^\[LATENCY\] role=(\S+) label=(\S+) duration_s=([\d.]+)"
)
LLM_CALLS_RE = re.compile(r"llm_calls=(\d+)")
RUNTIME_RE = re.compile(r"^\[runtime\] name=(\S+) seconds=([\d.]+)")

RUNS = {
    "1.7b-nt": "hotpotqa-selector-1.7b-nt",
    "1.7b-t": "hotpotqa-selector-1.7b-t",
    "4b-nt": "hotpotqa-selector-4b-nt",
    "4b-t": "hotpotqa-selector-4b-t",
    "8b-nt": "hotpotqa-selector-8b-nt",
    "8b-t": "hotpotqa-selector-8b-t",
    "14b-nt": "hotpotqa-selector-14b-nt",
    "14b-t": "hotpotqa-selector-14b-t",
    "gemma4e2b-nt": "hotpotqa-selector-gemma4e2b-nt",
    "gemma4e2b-t": "hotpotqa-selector-gemma4e2b-t",
}

AGENTS = ("WebSurfer", "Assistant", "ComputerTerminal")


def parse_first_round(path: Path) -> dict | None:
    """Return first-round latency dict, or None if rep had no [runtime]."""
    if not path.exists():
        return None

    sel_s = 0.0
    sel_calls = 0
    term_s = 0.0
    term_calls = 0
    agent_s = None
    agent_name = None
    agent_llm_s = 0.0
    agent_llm_calls = 0
    phase = "pre_runtime"  # pre_runtime -> post_runtime -> done

    with path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("[LATENCY]"):
                m = LATENCY_RE.match(line)
                if not m:
                    continue
                role, label, dur = m.group(1), m.group(2), float(m.group(3))
                if role == "orchestrator" and label == "selector":
                    if phase == "pre_runtime":
                        sel_s += dur
                        sel_calls += 1
                    elif phase == "post_runtime":
                        # next round has begun - stop
                        phase = "done"
                        break
                elif role == "orchestrator" and label == "termination":
                    if phase == "post_runtime":
                        term_s += dur
                        term_calls += 1
                elif role in ("web_surfer", "assistant") and phase == "pre_runtime":
                    # Agent's own LLM call(s) within this turn. Split out so
                    # tool_s = runtime - llm_s reveals browser/tool time.
                    agent_llm_s += dur
                    n = LLM_CALLS_RE.search(line)
                    agent_llm_calls += int(n.group(1)) if n else 1
            elif line.startswith("[runtime]"):
                m = RUNTIME_RE.match(line)
                if not m:
                    continue
                if phase == "pre_runtime":
                    agent_name = m.group(1)
                    agent_s = float(m.group(2))
                    phase = "post_runtime"

    if agent_s is None:
        return None

    tool_s = max(agent_s - agent_llm_s, 0.0)
    return {
        "agent": agent_name,
        "sel_s": sel_s,
        "agent_s": agent_s,
        "agent_llm_s": agent_llm_s,
        "tool_s": tool_s,
        "term_s": term_s,
        "round_e2e_s": sel_s + agent_s + term_s,
        "sel_calls": sel_calls,
        "term_calls": term_calls,
        "agent_llm_calls": agent_llm_calls,
    }


def analyze_run(run_dir: Path) -> dict:
    result = json.load((run_dir / "result.json").open())
    scenario_subdir = run_dir / result["scenario_name"]

    rows = []
    for task_id, task in result["tasks"].items():
        for rep_id, rep in task["repetitions"].items():
            if rep["status"] != "completed":
                continue
            log_path = scenario_subdir / task_id / str(rep_id) / "console_log.txt"
            r = parse_first_round(log_path)
            if r is not None:
                rows.append(r)

    n = len(rows)

    def mean(xs):
        return statistics.mean(xs) if xs else float("nan")

    agg = {
        "n_reps": n,
        "sel_s": mean([r["sel_s"] for r in rows]),
        "agent_s": mean([r["agent_s"] for r in rows]),
        "agent_llm_s": mean([r["agent_llm_s"] for r in rows]),
        "tool_s": mean([r["tool_s"] for r in rows]),
        "term_s": mean([r["term_s"] for r in rows]),
        "round_e2e_s": mean([r["round_e2e_s"] for r in rows]),
        "sel_calls": mean([r["sel_calls"] for r in rows]),
        "term_calls": mean([r["term_calls"] for r in rows]),
        "agent_llm_calls": mean([r["agent_llm_calls"] for r in rows]),
    }

    for a in AGENTS:
        sub = [r for r in rows if r["agent"] == a]
        agg[f"n_{a}"] = len(sub)
        agg[f"pct_{a}"] = (len(sub) / n) if n else 0.0
        agg[f"sel_s__{a}"] = mean([r["sel_s"] for r in sub])
        agg[f"agent_s__{a}"] = mean([r["agent_s"] for r in sub])
        agg[f"agent_llm_s__{a}"] = mean([r["agent_llm_s"] for r in sub])
        agg[f"tool_s__{a}"] = mean([r["tool_s"] for r in sub])
        agg[f"term_s__{a}"] = mean([r["term_s"] for r in sub])

    return agg


def main():
    rows = []
    for label, name in RUNS.items():
        run_root = OUT / name / "Results"
        if not run_root.exists():
            print(f"SKIP {label}: {run_root} missing")
            continue
        scenario_dirs = sorted(run_root.glob("hotpotqa_validation_fullwiki*"))
        if not scenario_dirs:
            print(f"SKIP {label}: no scenario dir")
            continue
        print(f"Processing {label} ({scenario_dirs[0].name}) ...")
        row = {"run": label}
        row.update(analyze_run(scenario_dirs[0]))
        rows.append(row)

    if not rows:
        print("No runs found")
        return

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {CSV_OUT}")

    print("\n=== Round-1 agent mix ===")
    print(
        f"{'run':<14} {'n':>4}  "
        f"{'WebSurf':>8} {'Assist':>7} {'CompTerm':>9}  "
        f"{'%WS':>5} {'%AS':>5} {'%CT':>5}"
    )
    for r in rows:
        n = r["n_reps"]
        print(
            f"{r['run']:<14} {n:>4}  "
            f"{r['n_WebSurfer']:>8d} {r['n_Assistant']:>7d} {r['n_ComputerTerminal']:>9d}  "
            f"{r['pct_WebSurfer']*100:>4.0f}% {r['pct_Assistant']*100:>4.0f}% "
            f"{r['pct_ComputerTerminal']*100:>4.0f}%"
        )

    print("\n=== First-round latency (avg over all reps) ===")
    print(
        f"{'run':<14} {'sel_s':>7} {'ag_llm':>7} {'tool_s':>7} "
        f"{'agent_s':>8} {'term_s':>7} {'e2e_s':>7}  "
        f"{'selN':>5} {'agN':>4} {'trmN':>5}"
    )
    for r in rows:
        print(
            f"{r['run']:<14} "
            f"{r['sel_s']:>7.2f} {r['agent_llm_s']:>7.2f} {r['tool_s']:>7.2f} "
            f"{r['agent_s']:>8.2f} {r['term_s']:>7.2f} {r['round_e2e_s']:>7.2f}  "
            f"{r['sel_calls']:>5.2f} {r['agent_llm_calls']:>4.2f} "
            f"{r['term_calls']:>5.2f}"
        )

    has_other = any(r["n_Assistant"] + r["n_ComputerTerminal"] > 0 for r in rows)
    if has_other:
        print("\n=== Conditional on round-1 agent (avg s; '-' if 0 reps) ===")
        print(
            f"{'run':<14} "
            f"{'WS_sel':>7} {'WS_agt':>7} {'WS_trm':>7}  "
            f"{'AS_sel':>7} {'AS_agt':>7} {'AS_trm':>7}  "
            f"{'CT_sel':>7} {'CT_agt':>7} {'CT_trm':>7}"
        )

        def fmt(v):
            return f"{v:>7.2f}" if v == v else f"{'-':>7}"  # NaN check

        for r in rows:
            print(
                f"{r['run']:<14} "
                f"{fmt(r['sel_s__WebSurfer'])} {fmt(r['agent_s__WebSurfer'])} "
                f"{fmt(r['term_s__WebSurfer'])}  "
                f"{fmt(r['sel_s__Assistant'])} {fmt(r['agent_s__Assistant'])} "
                f"{fmt(r['term_s__Assistant'])}  "
                f"{fmt(r['sel_s__ComputerTerminal'])} "
                f"{fmt(r['agent_s__ComputerTerminal'])} "
                f"{fmt(r['term_s__ComputerTerminal'])}"
            )


if __name__ == "__main__":
    main()
