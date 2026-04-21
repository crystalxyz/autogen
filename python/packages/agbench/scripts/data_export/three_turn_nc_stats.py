#!/usr/bin/env python3
"""Compute p50, p90, mean latency, pass/fail ratios, and per-turn LLM latency for each three-turn-nc setup.

Reads directly from outputs/ result.json and console_log.txt files.
"""

import csv
import glob
import json
import os
import re
import statistics

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "csv", "three_turn_nc_stats.csv")

RUNTIME_RE = re.compile(r"\[runtime\] name=(coder_turn\d+|executor_turn\d+) seconds=([\d.]+)")

FIELDNAMES = [
    "setup", "count", "accuracy",
    "pass_turn1_ratio", "pass_turn2_ratio", "pass_turn3_ratio", "fail_ratio",
    "mean_latency", "p50_latency", "p90_latency",
    "mean_llm_latency_turn1", "mean_llm_latency_turn2", "mean_llm_latency_turn3",
    "mean_e2e_pass_turn1", "mean_e2e_pass_turn2", "mean_e2e_pass_turn3", "mean_e2e_fail",
]


def parse_console_log(log_path):
    """Extract coder and executor runtimes from a console log.

    Returns dict mapping runtime name to seconds, e.g.
    {"coder_turn1": 2.1, "executor_turn1": 0.13, "coder_turn2": 3.5, ...}
    """
    runtimes = {}
    try:
        with open(log_path) as f:
            for line in f:
                m = RUNTIME_RE.search(line)
                if m:
                    runtimes[m.group(1)] = float(m.group(2))
    except OSError:
        pass
    return runtimes


def classify_turn(runtimes):
    """Determine which coding turn the task resolved at based on log runtimes.

    Returns 1, 2, or 3 depending on the highest coder_turnN or executor_turnN seen.
    """
    max_turn = 0
    for name in runtimes:
        m = re.search(r"turn(\d+)", name)
        if m:
            max_turn = max(max_turn, int(m.group(1)))
    return max_turn if max_turn > 0 else None


def main():
    setup_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "three-turn-*")))
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

        # Find the scenario subdirectory
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
        turn3_pass = 0
        failed = 0
        llm_latency_turn1 = []
        llm_latency_turn2 = []
        llm_latency_turn3 = []
        e2e_pass_turn1 = []
        e2e_pass_turn2 = []
        e2e_pass_turn3 = []
        e2e_fail = []

        for task_id, task_data in data["tasks"].items():
            for rep_id, rep_data in task_data["repetitions"].items():
                elapsed = rep_data.get("elapsed_time")
                if elapsed is None or elapsed == "":
                    continue
                elapsed = float(elapsed)
                times.append(elapsed)

                success = rep_data.get("success", False)

                # Parse console log for runtimes and turn classification
                log_path = os.path.join(
                    results_subdir, scenario_subdir, task_id, str(rep_id), "console_log.txt"
                )
                runtimes = parse_console_log(log_path)
                coding_turn = classify_turn(runtimes)

                if success:
                    if coding_turn == 1:
                        turn1_pass += 1
                        e2e_pass_turn1.append(elapsed)
                    elif coding_turn == 2:
                        turn2_pass += 1
                        e2e_pass_turn2.append(elapsed)
                    else:
                        turn3_pass += 1
                        e2e_pass_turn3.append(elapsed)
                else:
                    failed += 1
                    e2e_fail.append(elapsed)

                # Collect per-turn coder latencies
                if "coder_turn1" in runtimes:
                    llm_latency_turn1.append(runtimes["coder_turn1"])
                if "coder_turn2" in runtimes:
                    llm_latency_turn2.append(runtimes["coder_turn2"])
                if "coder_turn3" in runtimes:
                    llm_latency_turn3.append(runtimes["coder_turn3"])

        if not times:
            continue

        n = len(times)
        total = turn1_pass + turn2_pass + turn3_pass + failed
        times_sorted = sorted(times)

        def safe_mean(lst):
            return round(statistics.mean(lst), 2) if lst else ""

        all_rows.append({
            "setup": setup_name,
            "count": n,
            "accuracy": round((turn1_pass + turn2_pass + turn3_pass) / total, 4),
            "pass_turn1_ratio": round(turn1_pass / total, 4),
            "pass_turn2_ratio": round(turn2_pass / total, 4),
            "pass_turn3_ratio": round(turn3_pass / total, 4),
            "fail_ratio": round(failed / total, 4),
            "mean_latency": round(statistics.mean(times), 2),
            "p50_latency": round(statistics.median(times), 2),
            "p90_latency": round(times_sorted[int(n * 0.9)], 2),
            "mean_llm_latency_turn1": safe_mean(llm_latency_turn1),
            "mean_llm_latency_turn2": safe_mean(llm_latency_turn2),
            "mean_llm_latency_turn3": safe_mean(llm_latency_turn3),
            "mean_e2e_pass_turn1": safe_mean(e2e_pass_turn1),
            "mean_e2e_pass_turn2": safe_mean(e2e_pass_turn2),
            "mean_e2e_pass_turn3": safe_mean(e2e_pass_turn3),
            "mean_e2e_fail": safe_mean(e2e_fail),
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} setups to {OUTPUT_CSV}")
    header = (
        f"{'setup':<40} {'count':>5} {'acc':>6} {'t1%':>6} {'t2%':>6} {'t3%':>6} {'fail%':>6}"
        f" {'mean':>7} {'p50':>7} {'p90':>7}"
        f" {'llm_t1':>7} {'llm_t2':>7} {'llm_t3':>7}"
        f" {'e2e_t1':>7} {'e2e_t2':>7} {'e2e_t3':>7} {'e2e_f':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in all_rows:
        def fmt(val):
            return f"{val:>7.2f}" if val != "" else "    N/A"
        print(
            f"{r['setup']:<40} {r['count']:>5} {r['accuracy']:>6.4f}"
            f" {r['pass_turn1_ratio']:>6.4f} {r['pass_turn2_ratio']:>6.4f}"
            f" {r['pass_turn3_ratio']:>6.4f} {r['fail_ratio']:>6.4f}"
            f" {r['mean_latency']:>7.2f} {r['p50_latency']:>7.2f} {r['p90_latency']:>7.2f}"
            f" {fmt(r['mean_llm_latency_turn1'])} {fmt(r['mean_llm_latency_turn2'])}"
            f" {fmt(r['mean_llm_latency_turn3'])}"
            f" {fmt(r['mean_e2e_pass_turn1'])} {fmt(r['mean_e2e_pass_turn2'])}"
            f" {fmt(r['mean_e2e_pass_turn3'])} {fmt(r['mean_e2e_fail'])}"
        )


if __name__ == "__main__":
    main()
