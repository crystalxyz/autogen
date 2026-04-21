"""Generate task_min_model.csv: for each task, find the cheapest model with
first-turn pass rate > 0.5.

Models are ordered by capability (cheapest first):
  0.6b_nothink → 1.7b_nothink → 4b_nothink → 0.6b_think →
  8b_nothink → 1.7b_think → 4b_think → 8b_think

Usage:
    cd python/packages/agbench/outputs
    python ../scripts/task_min_model.py
"""

import csv
import sys

# Capability order (cheapest → most expensive)
MODEL_ORDER = [
    "0.6b_nothink", "1.7b_nothink", "4b_nothink", "0.6b_think",
    "8b_nothink", "1.7b_think", "4b_think", "8b_think",
]

FIRST_TURN_CSV = "../reports/task_pass_rates_first_turn.csv"
OUTPUT_CSV = "../reports/task_min_model.csv"
THRESHOLD = 0.5


def main():
    tasks = []
    rates = {}

    with open(FIRST_TURN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["Task"]
            tasks.append(task)
            for model in MODEL_ORDER:
                val = row.get(model, "")
                rates[(task, model)] = float(val) if val else 0.0

    rows = []
    counts = {}
    for model in MODEL_ORDER:
        counts[model] = 0
    counts["none"] = 0

    for task in tasks:
        min_model = "none"
        for model in MODEL_ORDER:
            if rates[(task, model)] > THRESHOLD:
                min_model = model
                break
        rows.append((task, min_model))
        counts[min_model] += 1

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task", "min_model"])
        for task, model in rows:
            w.writerow([task, model])

    print(f"Min-model distribution (first-turn, threshold={THRESHOLD}):")
    for model in MODEL_ORDER:
        if counts[model] > 0:
            print(f"  {model:16s}: {counts[model]:3d}")
    if counts["none"] > 0:
        print(f"  {'none':16s}: {counts['none']:3d}")
    print(f"\nCSV saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
