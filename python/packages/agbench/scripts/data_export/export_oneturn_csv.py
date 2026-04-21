#!/usr/bin/env python3
"""Export one-turn baseline results to CSV.

Reads all outputs/one-turn-* runs and produces a single CSV with one row per
setup containing success rate and e2e latency statistics (p50, p90, mean).

Usage:
    python scripts/export_oneturn_csv.py [-o reports/csv/one_turn_baselines.csv]
"""

import argparse
import glob
import json
import os
import re
import statistics

import numpy as np

RUNTIME_RE = re.compile(r"\[runtime\] name=coder\S* seconds=([\d.]+)")


def parse_turn1_runtime(log_path: str):
    try:
        with open(log_path) as f:
            for line in f:
                m = RUNTIME_RE.search(line)
                if m:
                    return float(m.group(1))
    except OSError:
        return None
    return None


def parse_run_name(name: str):
    """Parse 'one-turn-{size}-{mode}' into (model_size, mode)."""
    # e.g. one-turn-0.6b-nothink -> ('0.6b', 'nothink')
    parts = name.split("-", 2)  # ['one', 'turn', '0.6b-nothink']
    rest = parts[2]  # '0.6b-nothink' or '14b-think'
    # Split on last hyphen
    idx = rest.rfind("-")
    return rest[:idx], rest[idx + 1 :]


def load_run(run_dir: str):
    """Load result.json and compute per-rep latency stats."""
    rj_files = glob.glob(os.path.join(run_dir, "Results", "*", "result.json"))
    if not rj_files:
        return None
    result_file = rj_files[0]
    results_subdir = os.path.dirname(result_file)
    with open(result_file) as f:
        data = json.load(f)

    scenario_dirs = [
        d for d in os.listdir(results_subdir)
        if os.path.isdir(os.path.join(results_subdir, d))
    ]
    scenario_subdir = scenario_dirs[0] if scenario_dirs else None

    latencies = []
    pass_latencies = []
    fail_latencies = []
    llm_turn1 = []
    for task_id, task in data["tasks"].items():
        for rep_id, rep in task["repetitions"].items():
            if rep["status"] != "completed":
                continue
            lat = rep["elapsed_time"]
            latencies.append(lat)
            if rep.get("success"):
                pass_latencies.append(lat)
            else:
                fail_latencies.append(lat)
            if scenario_subdir is not None:
                log_path = os.path.join(
                    results_subdir, scenario_subdir, task_id, rep_id, "console_log.txt"
                )
                t1 = parse_turn1_runtime(log_path)
                if t1 is not None:
                    llm_turn1.append(t1)

    stats = data["stats"]
    return {
        "success_rate": stats["success_rate"],
        "total_reps": stats["total_repetitions"],
        "completed_reps": stats["completed_repetitions"],
        "latency_mean": round(statistics.mean(latencies), 2),
        "latency_p50": round(float(np.percentile(latencies, 50)), 2),
        "latency_p90": round(float(np.percentile(latencies, 90)), 2),
        "mean_e2e_pass_turn1": round(statistics.mean(pass_latencies), 2) if pass_latencies else "",
        "mean_e2e_fail": round(statistics.mean(fail_latencies), 2) if fail_latencies else "",
        "mean_llm_latency_turn1": round(statistics.mean(llm_turn1), 2) if llm_turn1 else "",
    }


def main():
    parser = argparse.ArgumentParser(description="Export one-turn baselines to CSV")
    parser.add_argument(
        "-o",
        "--output",
        default="reports/csv/one_turn_baselines.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
    run_dirs = sorted(glob.glob(os.path.join(outputs_dir, "one-turn-*")))

    # Sort by model size numerically
    size_order = {"0.6b": 0, "1.7b": 1, "4b": 2, "8b": 3, "14b": 4}

    rows = []
    for run_dir in run_dirs:
        name = os.path.basename(run_dir)
        model_size, mode = parse_run_name(name)
        result = load_run(run_dir)
        if result is None:
            print(f"WARNING: no result.json found for {name}, skipping")
            continue
        rows.append((model_size, mode, name, result))

    rows.sort(key=lambda r: (size_order.get(r[0], 99), r[1]))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        header = "setup,model_size,mode,success_rate,total_reps,completed_reps,latency_mean,latency_p50,latency_p90,mean_llm_latency_turn1,mean_e2e_pass_turn1,mean_e2e_fail"
        f.write(header + "\n")
        for model_size, mode, name, r in rows:
            f.write(
                f"{name},{model_size},{mode},{r['success_rate']},{r['total_reps']},"
                f"{r['completed_reps']},{r['latency_mean']},{r['latency_p50']},"
                f"{r['latency_p90']},{r['mean_llm_latency_turn1']},"
                f"{r['mean_e2e_pass_turn1']},{r['mean_e2e_fail']}\n"
            )

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
