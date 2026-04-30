#!/usr/bin/env python3
"""Compute p50, p90, mean latency, pass/fail ratios, and per-turn LLM latency for each two-turn-nc setup.

Reads directly from outputs/ result.json and console_log.txt files.
"""

import csv
import glob
import json
import os
import re
import statistics

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "csv", "two_turn_nc_stats.csv")

RUNTIME_RE = re.compile(r"\[runtime\] name=coder\S* seconds=([\d.]+)")

FIELDNAMES = [
    "setup", "count", "accuracy",
    "pass_turn1_ratio", "pass_turn2_ratio", "fail_ratio",
    "mean_latency", "p50_latency", "p90_latency",
    "mean_llm_latency_turn1", "mean_llm_latency_turn2",
    "mean_e2e_pass_turn1", "mean_e2e_pass_turn2", "mean_e2e_fail",
]


def parse_console_log(log_path):
    """Extract coder runtimes from a console log. Returns (turn1_seconds, turn2_seconds|None)."""
    coder_times = []
    try:
        with open(log_path) as f:
            for line in f:
                m = RUNTIME_RE.search(line)
                if m:
                    coder_times.append(float(m.group(1)))
    except OSError:
        return None, None
    turn1 = coder_times[0] if len(coder_times) >= 1 else None
    turn2 = coder_times[1] if len(coder_times) >= 2 else None
    return turn1, turn2


def main():
    setup_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "two-turn-nc-*")))
    setup_dirs = [d for d in setup_dirs if not d.endswith("-backup")]

    all_rows = []
    for setup_dir in setup_dirs:
        setup_name = os.path.basename(setup_dir)
        result_files = glob.glob(os.path.join(setup_dir, "Results", "*", "result.json"))
        if not result_files:
            continue

        result_file = result_files[0]
        results_subdir = os.path.dirname(result_file)
        with open(result_file) as f:
            data = json.load(f)

        # Find the scenario subdirectory (e.g. human_eval_AgentChatCopyNoCtx)
        scenario_dirs = [
            d for d in os.listdir(results_subdir)
            if os.path.isdir(os.path.join(results_subdir, d))
        ]
        if not scenario_dirs:
            continue
        scenario_subdir = scenario_dirs[0]

        times = []
        turn1_pass = 0
        turn2_pass = 0
        failed = 0
        llm_latency_turn1 = []
        llm_latency_turn2 = []
        e2e_pass_turn1 = []
        e2e_pass_turn2 = []
        e2e_fail = []

        for task_id, task_data in data["tasks"].items():
            for rep_id, rep_data in task_data["repetitions"].items():
                elapsed = rep_data.get("elapsed_time")
                if elapsed is None or elapsed == "":
                    continue
                times.append(float(elapsed))

                success = rep_data.get("success", False)
                # Two-turn-nc setups define "round" as one coder->executor cycle;
                # rounds == 1 means the task passed on the first coder turn.
                rounds = rep_data.get("rounds") or 0

                if success and rounds <= 1:
                    turn1_pass += 1
                    e2e_pass_turn1.append(float(elapsed))
                elif success:
                    turn2_pass += 1
                    e2e_pass_turn2.append(float(elapsed))
                else:
                    failed += 1
                    e2e_fail.append(float(elapsed))

                # Parse console log for LLM latencies
                log_path = os.path.join(
                    results_subdir, scenario_subdir, task_id, str(rep_id), "console_log.txt"
                )
                t1, t2 = parse_console_log(log_path)
                if t1 is not None:
                    llm_latency_turn1.append(t1)
                if t2 is not None:
                    llm_latency_turn2.append(t2)

        if not times:
            continue

        n = len(times)
        total = turn1_pass + turn2_pass + failed
        times_sorted = sorted(times)

        all_rows.append({
            "setup": setup_name,
            "count": n,
            "accuracy": round((turn1_pass + turn2_pass) / total, 4),
            "pass_turn1_ratio": round(turn1_pass / total, 4),
            "pass_turn2_ratio": round(turn2_pass / total, 4),
            "fail_ratio": round(failed / total, 4),
            "mean_latency": round(statistics.mean(times), 2),
            "p50_latency": round(statistics.median(times), 2),
            "p90_latency": round(times_sorted[int(n * 0.9)], 2),
            "mean_llm_latency_turn1": round(statistics.mean(llm_latency_turn1), 2) if llm_latency_turn1 else "",
            "mean_llm_latency_turn2": round(statistics.mean(llm_latency_turn2), 2) if llm_latency_turn2 else "",
            "mean_e2e_pass_turn1": round(statistics.mean(e2e_pass_turn1), 2) if e2e_pass_turn1 else "",
            "mean_e2e_pass_turn2": round(statistics.mean(e2e_pass_turn2), 2) if e2e_pass_turn2 else "",
            "mean_e2e_fail": round(statistics.mean(e2e_fail), 2) if e2e_fail else "",
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} setups to {OUTPUT_CSV}")
    header = (
        f"{'setup':<30} {'count':>5} {'acc':>6} {'t1%':>6} {'t2%':>6} {'fail%':>6}"
        f" {'mean':>7} {'p50':>7} {'p90':>7} {'llm_t1':>7} {'llm_t2':>7}"
        f" {'e2e_t1':>7} {'e2e_t2':>7} {'e2e_f':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in all_rows:
        def fmt(val):
            return f"{val:>7.2f}" if val != "" else "    N/A"
        print(
            f"{r['setup']:<30} {r['count']:>5} {r['accuracy']:>6.4f} {r['pass_turn1_ratio']:>6.4f}"
            f" {r['pass_turn2_ratio']:>6.4f} {r['fail_ratio']:>6.4f}"
            f" {r['mean_latency']:>7.2f} {r['p50_latency']:>7.2f} {r['p90_latency']:>7.2f}"
            f" {fmt(r['mean_llm_latency_turn1'])} {fmt(r['mean_llm_latency_turn2'])}"
            f" {fmt(r['mean_e2e_pass_turn1'])} {fmt(r['mean_e2e_pass_turn2'])} {fmt(r['mean_e2e_fail'])}"
        )


if __name__ == "__main__":
    main()
