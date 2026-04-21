#!/usr/bin/env python3
"""Compute p50, p90, mean latency, pass/fail ratios, and per-turn LLM latency for each four-turn-nc setup.

Reads directly from outputs/ result.json and console_log.txt files.
"""

import csv
import glob
import json
import os
import re
import statistics

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "csv", "four_turn_nc_stats.csv")

RUNTIME_RE = re.compile(r"\[runtime\] name=(coder_turn\d+|executor_turn\d+) seconds=([\d.]+)")

FIELDNAMES = [
    "setup", "count", "accuracy",
    "pass_turn1_ratio", "pass_turn2_ratio", "pass_turn3_ratio", "pass_turn4_ratio", "fail_ratio",
    "mean_latency", "p50_latency", "p90_latency",
    "mean_llm_latency_turn1", "mean_llm_latency_turn2", "mean_llm_latency_turn3", "mean_llm_latency_turn4",
    "mean_e2e_pass_turn1", "mean_e2e_pass_turn2", "mean_e2e_pass_turn3", "mean_e2e_pass_turn4", "mean_e2e_fail",
]


def parse_console_log(log_path):
    """Extract coder and executor runtimes from a console log."""
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
    """Determine which coding turn the task resolved at based on log runtimes."""
    max_turn = 0
    for name in runtimes:
        m = re.search(r"turn(\d+)", name)
        if m:
            max_turn = max(max_turn, int(m.group(1)))
    return max_turn if max_turn > 0 else None


def main():
    setup_dirs = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "four-turn-nc-*")))
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

        scenario_dirs = [
            d for d in os.listdir(results_subdir)
            if os.path.isdir(os.path.join(results_subdir, d))
        ]
        if not scenario_dirs:
            continue
        scenario_subdir = scenario_dirs[0]

        times = []
        turn_pass = {1: 0, 2: 0, 3: 0, 4: 0}
        failed = 0
        llm_latency = {1: [], 2: [], 3: [], 4: []}
        e2e_pass = {1: [], 2: [], 3: [], 4: []}
        e2e_fail = []

        for task_id, task_data in data["tasks"].items():
            for rep_id, rep_data in task_data["repetitions"].items():
                elapsed = rep_data.get("elapsed_time")
                if elapsed is None or elapsed == "":
                    continue
                elapsed = float(elapsed)
                times.append(elapsed)

                success = rep_data.get("success", False)

                log_path = os.path.join(
                    results_subdir, scenario_subdir, task_id, str(rep_id), "console_log.txt"
                )
                runtimes = parse_console_log(log_path)
                coding_turn = classify_turn(runtimes)

                if success:
                    t = coding_turn if coding_turn in (1, 2, 3, 4) else 4
                    turn_pass[t] += 1
                    e2e_pass[t].append(elapsed)
                else:
                    failed += 1
                    e2e_fail.append(elapsed)

                for t in range(1, 5):
                    key = f"coder_turn{t}"
                    if key in runtimes:
                        llm_latency[t].append(runtimes[key])

        if not times:
            continue

        n = len(times)
        total = sum(turn_pass.values()) + failed
        times_sorted = sorted(times)

        def safe_mean(lst):
            return round(statistics.mean(lst), 2) if lst else ""

        all_rows.append({
            "setup": setup_name,
            "count": n,
            "accuracy": round(sum(turn_pass.values()) / total, 4),
            "pass_turn1_ratio": round(turn_pass[1] / total, 4),
            "pass_turn2_ratio": round(turn_pass[2] / total, 4),
            "pass_turn3_ratio": round(turn_pass[3] / total, 4),
            "pass_turn4_ratio": round(turn_pass[4] / total, 4),
            "fail_ratio": round(failed / total, 4),
            "mean_latency": round(statistics.mean(times), 2),
            "p50_latency": round(statistics.median(times), 2),
            "p90_latency": round(times_sorted[int(n * 0.9)], 2),
            "mean_llm_latency_turn1": safe_mean(llm_latency[1]),
            "mean_llm_latency_turn2": safe_mean(llm_latency[2]),
            "mean_llm_latency_turn3": safe_mean(llm_latency[3]),
            "mean_llm_latency_turn4": safe_mean(llm_latency[4]),
            "mean_e2e_pass_turn1": safe_mean(e2e_pass[1]),
            "mean_e2e_pass_turn2": safe_mean(e2e_pass[2]),
            "mean_e2e_pass_turn3": safe_mean(e2e_pass[3]),
            "mean_e2e_pass_turn4": safe_mean(e2e_pass[4]),
            "mean_e2e_fail": safe_mean(e2e_fail),
        })

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} setups to {OUTPUT_CSV}")
    header = (
        f"{'setup':<50} {'count':>5} {'acc':>6} {'t1%':>6} {'t2%':>6} {'t3%':>6} {'t4%':>6} {'fail%':>6}"
        f" {'mean':>7} {'p50':>7} {'p90':>7}"
        f" {'llm_t1':>7} {'llm_t2':>7} {'llm_t3':>7} {'llm_t4':>7}"
        f" {'e2e_t1':>7} {'e2e_t2':>7} {'e2e_t3':>7} {'e2e_t4':>7} {'e2e_f':>7}"
    )
    print(f"\n{header}")
    print("-" * len(header))
    for r in all_rows:
        def fmt(val):
            return f"{val:>7.2f}" if val != "" else "    N/A"
        print(
            f"{r['setup']:<50} {r['count']:>5} {r['accuracy']:>6.4f}"
            f" {r['pass_turn1_ratio']:>6.4f} {r['pass_turn2_ratio']:>6.4f}"
            f" {r['pass_turn3_ratio']:>6.4f} {r['pass_turn4_ratio']:>6.4f}"
            f" {r['fail_ratio']:>6.4f}"
            f" {r['mean_latency']:>7.2f} {r['p50_latency']:>7.2f} {r['p90_latency']:>7.2f}"
            f" {fmt(r['mean_llm_latency_turn1'])} {fmt(r['mean_llm_latency_turn2'])}"
            f" {fmt(r['mean_llm_latency_turn3'])} {fmt(r['mean_llm_latency_turn4'])}"
            f" {fmt(r['mean_e2e_pass_turn1'])} {fmt(r['mean_e2e_pass_turn2'])}"
            f" {fmt(r['mean_e2e_pass_turn3'])} {fmt(r['mean_e2e_pass_turn4'])}"
            f" {fmt(r['mean_e2e_fail'])}"
        )


if __name__ == "__main__":
    main()
