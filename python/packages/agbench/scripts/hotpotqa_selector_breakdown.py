"""Aggregate HotpotQA selector runs: accuracy vs time + per-agent time/turn breakdown.

Parses:
- result.json: per-rep status, elapsed_time, turns, success
- console_log.txt: [LATENCY] lines (orchestrator selector/termination/finalizer,
  web_surfer, assistant LLM durations) and [runtime] lines (per-agent
  end-to-end seconds including non-LLM time like browser/code execution).

Per-run aggregate (averaged over completed repetitions) written to
reports/csv/hotpotqa_selector_breakdown.csv.
"""

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path("/home/xz957/autogen/python/packages/agbench")
OUT = ROOT / "outputs"
CSV_OUT = ROOT / "reports" / "csv" / "hotpotqa_selector_breakdown.csv"

LATENCY_RE = re.compile(
    r"^\[LATENCY\] role=(\S+) label=(\S+) duration_s=([\d.]+)"
    r"(?: prompt_tokens=(\d+))?(?: completion_tokens=(\d+))?"
)
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
}


def parse_console(path: Path) -> dict:
    """Return per-rep aggregates from one console_log.txt."""
    agg = {
        "orch_selector_s": 0.0,
        "orch_termination_s": 0.0,
        "orch_finalizer_s": 0.0,
        "orch_selector_calls": 0,
        "orch_termination_calls": 0,
        "orch_finalizer_calls": 0,
        "web_surfer_llm_s": 0.0,
        "web_surfer_llm_calls": 0,
        "assistant_llm_s": 0.0,
        "assistant_llm_calls": 0,
        "WebSurfer_runtime_s": 0.0,
        "WebSurfer_turns": 0,
        "Assistant_runtime_s": 0.0,
        "Assistant_turns": 0,
        "ComputerTerminal_runtime_s": 0.0,
        "ComputerTerminal_turns": 0,
    }
    if not path.exists():
        return agg
    with path.open("r", errors="replace") as fh:
        for line in fh:
            if line.startswith("[LATENCY]"):
                m = LATENCY_RE.match(line)
                if not m:
                    continue
                role, label, dur = m.group(1), m.group(2), float(m.group(3))
                if role == "orchestrator":
                    if label == "selector":
                        agg["orch_selector_s"] += dur
                        agg["orch_selector_calls"] += 1
                    elif label == "termination":
                        agg["orch_termination_s"] += dur
                        agg["orch_termination_calls"] += 1
                    elif label == "finalizer":
                        agg["orch_finalizer_s"] += dur
                        agg["orch_finalizer_calls"] += 1
                elif role == "web_surfer":
                    agg["web_surfer_llm_s"] += dur
                    agg["web_surfer_llm_calls"] += 1
                elif role == "assistant":
                    agg["assistant_llm_s"] += dur
                    agg["assistant_llm_calls"] += 1
            elif line.startswith("[runtime]"):
                m = RUNTIME_RE.match(line)
                if not m:
                    continue
                name, sec = m.group(1), float(m.group(2))
                key_rt = f"{name}_runtime_s"
                key_turns = f"{name}_turns"
                if key_rt in agg:
                    agg[key_rt] += sec
                    agg[key_turns] += 1
    return agg


def analyze_run(run_dir: Path) -> dict:
    result = json.load((run_dir / "result.json").open())
    scenario_subdir = run_dir / result["scenario_name"]

    total = 0
    succ = 0
    e2e_times = []
    turns_list = []

    # per-rep aggregates to average
    per_rep_rows = []
    for task_id, task in result["tasks"].items():
        for rep_id, rep in task["repetitions"].items():
            if rep["status"] != "completed":
                continue
            total += 1
            if rep["success"]:
                succ += 1
            e2e_times.append(rep["elapsed_time"])
            turns_list.append(rep["turns"])
            log_path = scenario_subdir / task_id / str(rep_id) / "console_log.txt"
            per_rep_rows.append(parse_console(log_path))

    def mean(xs):
        return statistics.mean(xs) if xs else 0.0

    keys = list(per_rep_rows[0].keys()) if per_rep_rows else []
    avg_breakdown = {k: mean([r[k] for r in per_rep_rows]) for k in keys}

    return {
        "completed_reps": total,
        "success_rate": succ / total if total else 0.0,
        "avg_e2e_s": mean(e2e_times),
        "median_e2e_s": statistics.median(e2e_times) if e2e_times else 0.0,
        "avg_turns": mean(turns_list),
        **avg_breakdown,
    }


def main():
    rows = []
    for label, name in RUNS.items():
        run_root = OUT / name / "Results"
        if not run_root.exists():
            print(f"SKIP {label}: {run_root} missing")
            continue
        # single scenario subdir
        scenario_dirs = sorted(run_root.glob("hotpotqa_validation_fullwiki*"))
        if not scenario_dirs:
            print(f"SKIP {label}: no scenario dir")
            continue
        print(f"Processing {label} ({scenario_dirs[0].name}) ...")
        row = {"run": label}
        row.update(analyze_run(scenario_dirs[0]))
        rows.append(row)

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with CSV_OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {CSV_OUT}")

    # Print concise tables
    def p(v, w=8, prec=2):
        if isinstance(v, float):
            return f"{v:>{w}.{prec}f}"
        return f"{str(v):>{w}}"

    print("\n=== Accuracy vs E2E time ===")
    print(f"{'run':<10} {'reps':>5} {'acc':>7} {'avgE2E':>8} {'medE2E':>8} {'avgTurns':>9}")
    for r in rows:
        print(
            f"{r['run']:<10} {r['completed_reps']:>5} "
            f"{r['success_rate']:>7.3f} {r['avg_e2e_s']:>8.1f} "
            f"{r['median_e2e_s']:>8.1f} {r['avg_turns']:>9.2f}"
        )

    print("\n=== Per-agent avg turns (msgs emitted) ===")
    print(f"{'run':<10} {'WebSurf':>8} {'Assist':>8} {'CodeExec':>9}")
    for r in rows:
        print(
            f"{r['run']:<10} {r['WebSurfer_turns']:>8.2f} "
            f"{r['Assistant_turns']:>8.2f} {r['ComputerTerminal_turns']:>9.2f}"
        )

    print("\n=== Per-agent avg runtime (s, incl. non-LLM) ===")
    print(f"{'run':<10} {'WebSurf':>8} {'Assist':>8} {'CodeExec':>9}")
    for r in rows:
        print(
            f"{r['run']:<10} {r['WebSurfer_runtime_s']:>8.1f} "
            f"{r['Assistant_runtime_s']:>8.1f} {r['ComputerTerminal_runtime_s']:>9.1f}"
        )

    print("\n=== Orchestrator LLM time (s) & calls ===")
    print(
        f"{'run':<10} {'selS':>6} {'selN':>5} {'termS':>7} "
        f"{'termN':>6} {'finS':>6} {'finN':>5}"
    )
    for r in rows:
        print(
            f"{r['run']:<10} "
            f"{r['orch_selector_s']:>6.1f} {r['orch_selector_calls']:>5.2f} "
            f"{r['orch_termination_s']:>7.1f} {r['orch_termination_calls']:>6.2f} "
            f"{r['orch_finalizer_s']:>6.1f} {r['orch_finalizer_calls']:>5.2f}"
        )

    print("\n=== Agent LLM-only time (s) & calls ===")
    print(f"{'run':<10} {'WS_llmS':>8} {'WS_N':>6} {'AS_llmS':>8} {'AS_N':>6}")
    for r in rows:
        print(
            f"{r['run']:<10} "
            f"{r['web_surfer_llm_s']:>8.1f} {r['web_surfer_llm_calls']:>6.2f} "
            f"{r['assistant_llm_s']:>8.1f} {r['assistant_llm_calls']:>6.2f}"
        )


if __name__ == "__main__":
    main()
