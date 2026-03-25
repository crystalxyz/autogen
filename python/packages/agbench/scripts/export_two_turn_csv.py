#!/usr/bin/env python3
"""Export all two-turn benchmark results into a single CSV."""

import csv
import glob
import json
import os
import sys

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "reports", "csv", "two_turn_results.csv")


def main():
    rows = []
    pattern = os.path.join(OUTPUTS_DIR, "two-turn-*")
    setup_dirs = sorted(glob.glob(pattern))

    for setup_dir in setup_dirs:
        setup_name = os.path.basename(setup_dir)
        result_files = glob.glob(os.path.join(setup_dir, "Results", "*", "result.json"))
        if not result_files:
            print(f"WARNING: No result.json found in {setup_dir}", file=sys.stderr)
            continue

        result_file = result_files[0]
        with open(result_file) as f:
            data = json.load(f)

        for task_id, task_data in data["tasks"].items():
            for rep_id, rep_data in task_data["repetitions"].items():
                rows.append({
                    "setup": setup_name,
                    "task_id": task_id,
                    "repetition": int(rep_id),
                    "success": rep_data["success"],
                    "time": rep_data["elapsed_time"],
                })

    # Sort by setup, task_id, repetition
    rows.sort(key=lambda r: (r["setup"], r["task_id"], r["repetition"]))

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["setup", "task_id", "repetition", "success", "time"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
