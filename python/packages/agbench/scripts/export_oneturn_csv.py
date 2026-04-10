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
import statistics

import numpy as np


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
    with open(rj_files[0]) as f:
        data = json.load(f)

    # Collect all completed rep latencies
    latencies = []
    for task in data["tasks"].values():
        for rep in task["repetitions"].values():
            if rep["status"] == "completed":
                latencies.append(rep["elapsed_time"])

    stats = data["stats"]
    return {
        "success_rate": stats["success_rate"],
        "total_reps": stats["total_repetitions"],
        "completed_reps": stats["completed_repetitions"],
        "latency_mean": round(statistics.mean(latencies), 2),
        "latency_p50": round(float(np.percentile(latencies, 50)), 2),
        "latency_p90": round(float(np.percentile(latencies, 90)), 2),
        "latency_min": round(min(latencies), 2),
        "latency_max": round(max(latencies), 2),
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

    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
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
        header = "setup,model_size,mode,success_rate,total_reps,completed_reps,latency_mean,latency_p50,latency_p90,latency_min,latency_max"
        f.write(header + "\n")
        for model_size, mode, name, r in rows:
            f.write(
                f"{name},{model_size},{mode},{r['success_rate']},{r['total_reps']},"
                f"{r['completed_reps']},{r['latency_mean']},{r['latency_p50']},"
                f"{r['latency_p90']},{r['latency_min']},{r['latency_max']}\n"
            )

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
