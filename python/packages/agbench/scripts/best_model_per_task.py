"""Compute the best (cheapest) model per task given first-turn pass rates and latencies.

For each task, picks the cheapest model (lowest first-turn latency) that achieves
the highest first-turn pass rate. Outputs a CSV and summary statistics.

Usage:
    cd python/packages/agbench/outputs
    python ../scripts/best_model_per_task.py
"""

import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

MODELS = [
    "0.6b_nothink", "1.7b_nothink", "4b_nothink", "8b_nothink",
    "0.6b_think", "1.7b_think", "4b_think", "8b_think",
]
OUTPUT_DIRS = {m: f"output_{m}" for m in MODELS}

FIRST_TURN_CSV = "../reports/task_pass_rates_first_turn.csv"
OUTPUT_CSV = "../reports/best_model_per_task.csv"


def load_first_turn_pass_rates(csv_path):
    """Load first-turn pass rates from CSV."""
    tasks = []
    rates = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["Task"]
            tasks.append(task)
            for model in MODELS:
                val = row.get(model, "")
                rates[(task, model)] = float(val) if val else 0.0
    return tasks, rates


def collect_first_turn_latencies(output_dirs):
    """Collect average first-turn coder latency per (task, model)."""
    latencies = defaultdict(list)
    for model, dirname in output_dirs.items():
        base = Path(dirname) / "Results" / "human_eval"
        agent_dirs = list(base.glob("human_eval_AgentChat*"))
        if not agent_dirs:
            continue
        agent_dir = agent_dirs[0]
        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("HumanEval_"):
                continue
            task = task_dir.name
            for trial_dir in sorted(task_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                log_path = trial_dir / "console_log.txt"
                if not log_path.exists():
                    continue
                text = log_path.read_text(errors="replace")
                match = re.search(r"\[runtime\]\s*name=coder\s+seconds=([\d.]+)", text)
                if match:
                    latencies[(task, model)].append(float(match.group(1)))
    return latencies


def main():
    tasks, pass_rates = load_first_turn_pass_rates(FIRST_TURN_CSV)
    latencies_raw = collect_first_turn_latencies(OUTPUT_DIRS)

    # Average latency per (task, model)
    avg_latency = {}
    for key, vals in latencies_raw.items():
        avg_latency[key] = np.mean(vals)

    rows = []
    model_counts = defaultdict(int)

    for task in tasks:
        best_model = None
        best_rate = -1.0
        best_lat = float("inf")

        for model in MODELS:
            rate = pass_rates.get((task, model), 0.0)
            lat = avg_latency.get((task, model), float("inf"))

            # Pick highest pass rate; break ties by lowest latency
            if rate > best_rate or (rate == best_rate and lat < best_lat):
                best_model = model
                best_rate = rate
                best_lat = lat

        rows.append((task, best_model, best_rate, best_lat))
        model_counts[best_model] += 1

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task", "Best_Model", "First_Turn_Pass_Rate", "Avg_First_Turn_Latency_s"])
        for task, model, rate, lat in rows:
            w.writerow([task, model, f"{rate:.3f}", f"{lat:.3f}"])

    # Summary
    total_pass = sum(r for _, _, r, _ in rows)
    total_lat = sum(l for _, _, _, l in rows)
    n = len(tasks)

    print(f"Best model per task (first-turn, {n} tasks):")
    print(f"  Total expected pass: {total_pass:.1f}/{n} ({total_pass/n*100:.1f}%)")
    print(f"  Total latency: {total_lat:.1f}s")
    print(f"\n  Model distribution:")
    for model in MODELS:
        if model_counts[model] > 0:
            print(f"    {model:16s}: {model_counts[model]:3d} tasks")

    # Compare with uniform baselines
    print(f"\n  Uniform baselines (first-turn):")
    for model in MODELS:
        rates = [pass_rates.get((t, model), 0.0) for t in tasks]
        lats = [avg_latency.get((t, model), 0.0) for t in tasks]
        print(f"    {model:16s}: pass={sum(rates):.1f}/{n} ({sum(rates)/n*100:.1f}%)  "
              f"lat={sum(lats):.1f}s")

    print(f"\nCSV saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
