"""Generate per-task pass rate tables across model sizes and think/nothink configs.

Produces two CSVs:
  1. task_pass_rates.csv         — overall pass rates (from result.json)
  2. task_pass_rates_first_turn.csv — first-turn pass rates (from console logs)

First-turn pass is determined by counting [runtime] name=coder entries in
each trial's console_log.txt.  Exactly 1 coder invocation means the first
attempt passed (no retry needed).

Usage:
    cd python/packages/agbench/outputs
    python ../scripts/task_pass_rates.py
"""

import json
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

DIRS = {
    "0.6b_think": "output_0.6b_think",
    "0.6b_nothink": "output_0.6b_nothink",
    "1.7b_think": "output_1.7b_think",
    "1.7b_nothink": "output_1.7b_nothink",
    "4b_think": "output_4b_think",
    "4b_nothink": "output_4b_nothink",
    "8b_think": "output_8b_think",
    "8b_nothink": "output_8b_nothink",
}

COL_ORDER = [
    "0.6b_nothink", "1.7b_nothink", "4b_nothink", "8b_nothink",
    "0.6b_think", "1.7b_think", "4b_think", "8b_think",
]

OUTPUT_CSV = "../reports/task_pass_rates.csv"
OUTPUT_CSV_FIRST_TURN = "../reports/task_pass_rates_first_turn.csv"


def _write_csv_and_print(path, header, rows, label):
    """Write a CSV and echo to stdout."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
    # stdout
    print(",".join(header))
    for row in rows:
        print(",".join(str(c) for c in row))
    print(f"\n{label} saved to {path}", file=sys.stderr)


def compute_overall_pass_rates():
    """Load overall pass rates from result.json files."""
    all_data = {}
    all_tasks = set()
    for label, d in DIRS.items():
        result_path = Path(d) / "Results" / "human_eval" / "result.json"
        try:
            with open(result_path) as f:
                data = json.load(f)
            all_data[label] = data["tasks"]
            all_tasks.update(data["tasks"].keys())
        except FileNotFoundError:
            print(f"Warning: {result_path} not found, skipping {label}", file=sys.stderr)

    sorted_tasks = sorted(all_tasks, key=lambda t: int(t.replace("HumanEval_", "")))
    rows = []
    for task in sorted_tasks:
        row = [task]
        for col in COL_ORDER:
            if col in all_data and task in all_data[col]:
                s = all_data[col][task]["stats"]
                rate = s["successful"] / s["total_repetitions"]
                row.append(f"{rate:.3f}")
            else:
                row.append("")
        rows.append(row)

    return sorted_tasks, rows


def compute_first_turn_pass_rates():
    """Compute first-turn pass rates by parsing console logs.

    First-turn passes iff exactly 1 [runtime] name=coder entry exists
    in the console log (no retry was needed).
    """
    # (task, model) -> list of bool
    ft_pass = defaultdict(list)
    all_tasks = set()

    for label, d in DIRS.items():
        base = Path(d) / "Results" / "human_eval"
        agent_dirs = list(base.glob("human_eval_AgentChat*"))
        if not agent_dirs:
            print(f"Warning: no AgentChat dir in {base}, skipping {label}", file=sys.stderr)
            continue
        agent_dir = agent_dirs[0]

        for task_dir in sorted(agent_dir.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("HumanEval_"):
                continue
            task = task_dir.name
            all_tasks.add(task)

            for trial_dir in sorted(task_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                log_path = trial_dir / "console_log.txt"
                if not log_path.exists():
                    continue
                text = log_path.read_text(errors="replace")
                n_coder = len(re.findall(
                    r"\[runtime\]\s*name=coder\s+seconds=[\d.]+", text
                ))
                if n_coder > 0:
                    ft_pass[(task, label)].append(n_coder == 1)

    sorted_tasks = sorted(all_tasks, key=lambda t: int(t.replace("HumanEval_", "")))
    rows = []
    for task in sorted_tasks:
        row = [task]
        for col in COL_ORDER:
            trials = ft_pass.get((task, col), [])
            if trials:
                rate = sum(trials) / len(trials)
                row.append(f"{rate:.3f}")
            else:
                row.append("")
        rows.append(row)

    return sorted_tasks, rows


def main():
    # Overall pass rates
    print("=== Overall pass rates ===")
    tasks_overall, rows_overall = compute_overall_pass_rates()
    _write_csv_and_print(OUTPUT_CSV, ["Task"] + COL_ORDER, rows_overall, "Overall pass rates")

    print()

    # First-turn pass rates
    print("=== First-turn pass rates ===")
    tasks_ft, rows_ft = compute_first_turn_pass_rates()
    _write_csv_and_print(OUTPUT_CSV_FIRST_TURN, ["Task"] + COL_ORDER, rows_ft, "First-turn pass rates")

    # Summary comparison
    print("\n=== Summary ===", file=sys.stderr)
    for j, model in enumerate(COL_ORDER):
        overall_vals = [float(r[j + 1]) for r in rows_overall if r[j + 1]]
        ft_vals = [float(r[j + 1]) for r in rows_ft if r[j + 1]]
        overall_avg = sum(overall_vals) / len(overall_vals) if overall_vals else 0
        ft_avg = sum(ft_vals) / len(ft_vals) if ft_vals else 0
        print(f"  {model:16s}  overall={overall_avg:.3f}  first_turn={ft_avg:.3f}  "
              f"delta={overall_avg - ft_avg:+.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
