#!/usr/bin/env python3
"""Draw a stacked bar chart comparing turn distributions across multiple runs.

Each bar shows the breakdown of repetitions by turn count (2, 4, 6, 8, 10, 12),
separating successful tasks from failed tasks (failed at 12 turns vs. failed early).

Usage:
    python compare_turn_distribution.py --run "label1" path1.json --run "label2" path2.json ...
    python compare_turn_distribution.py --run "label1" path1.json --run "label2" path2.json ... -o chart.png

Example (all output_* runs):
    python scripts/compare_turn_distribution.py \
      --run "0.6b_le1000" output_0.6b_1000output/Results/human_eval/result.json \
      --run "0.6b_eq1000" output_0.6b_eq1000output/Results/human_eval/result.json \
      --run "0.6b_le3000" output_0.6b_3000output/Results/human_eval/result.json \
      --run "0.6b_eq3000" output_0.6b_eq3000output/Results/human_eval/result.json \
      --run "0.6b_le5000" output_0.6b_5000output/Results/human_eval/result.json \
      --run "0.6b" output_0.6b/human_eval/result.json \
      --run "1.7b_le1000" output_1.7b_1000output/Results/human_eval/result.json \
      --run "1.7b_eq1000" output_1.7b_eq1000output/Results/human_eval/result.json \
      --run "1.7b_le3000" output_1.7b_3000output/Results/human_eval/result.json \
      --run "1.7b_eq3000" output_1.7b_eq3000output/Results/human_eval/result.json \
      --run "1.7b_le5000" output_1.7b_5000output/Results/human_eval/result.json \
      --run "1.7b" output_1.7b/human_eval/result.json \
      --run "4b_le1000" output_4b_1000output/Results/human_eval/result.json \
      --run "4b_eq1000" output_4b_eq1000output/Results/human_eval/result.json \
      --run "4b_le3000" output_4b_3000output/Results/human_eval/result.json \
      --run "4b_eq3000" output_4b_eq3000output/Results/human_eval/result.json \
      --run "4b_le5000" output_4b_5000output/Results/human_eval/result.json \
      --run "4b" output_4b/human_eval/result.json \
      --run "8b_le1000" output_8b_1000output/Results/human_eval/result.json \
      --run "8b_eq1000" output_8b_eq1000output/Results/human_eval/result.json \
      --run "8b_le3000" output_8b_3000output/Results/human_eval/result.json \
      --run "8b_eq3000" output_8b_eq3000output/Results/human_eval/result.json \
      --run "8b_le5000" output_8b_5000output/Results/human_eval/result.json \
      --run "8b" output_8b/human_eval/result.json \
      --normalize -o reports/turn_distribution_comparison.png

    python scripts/compare_turn_distribution.py \
      --run "0.6b-nothink" output_0.6b_nothink/Results/human_eval/result.json \
      --run "1.7b-nothink" output_1.7b_nothink/Results/human_eval/result.json \
      --run "4b-nothink" output_4b_nothink/Results/human_eval/result.json \
      --run "8b-nothink" output_8b_nothink/Results/human_eval/result.json \
      --run "0.6b" output_0.6b_think/Results/human_eval/result.json \
      --run "1.7b" output_1.7b_think/Results/human_eval/result.json \
      --run "4b" output_4b_think/Results/human_eval/result.json \
      --run "8b" output_8b_think/Results/human_eval/result.json \
      --normalize -o reports/accuracy_comp_think_vs_nothink.png

To handle corrupted JSON (e.g., broken lines), the script attempts a line-level fix.
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


TURN_BUCKETS = [2, 4, 6, 8, 10, 12]


def load_result(path: str) -> dict:
    """Load result.json, attempting to fix known corruption."""
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            pass

    # Attempt line-level fix for corrupted avg_turns_per_task line
    with open(path) as f:
        lines = f.readlines()
    fixed_lines = []
    for line in lines:
        if "avg_turns_per_task" in line and not line.strip().startswith('"avg_turns_per_task"'):
            # Corrupted line like: "": 2.28,avg_turns_per_task
            # Extract the value and rebuild
            import re

            m = re.search(r"([\d.]+)", line)
            val = m.group(1) if m else "0"
            indent = line[: len(line) - len(line.lstrip())]
            fixed_lines.append(f'{indent}"avg_turns_per_task": {val},\n')
        else:
            fixed_lines.append(line)
    return json.loads("".join(fixed_lines))


def count_by_turns(data: dict) -> dict:
    """Return counts dict with success_N_turns, failed_12_turns, and failed_early keys."""
    success_counts = {t: 0 for t in TURN_BUCKETS}
    other_success = 0
    failed_12 = 0
    failed_early = 0

    for task in data["tasks"].values():
        for rep in task["repetitions"].values():
            turns = rep.get("turns")
            if turns is None:
                failed_early += 1
                continue
            if rep["success"]:
                if turns in success_counts:
                    success_counts[turns] += 1
                else:
                    other_success += 1
            else:
                if turns >= 12:
                    failed_12 += 1
                else:
                    failed_early += 1

    result = {f"success_{t}": success_counts[t] for t in TURN_BUCKETS}
    if other_success > 0:
        result["success_other"] = other_success
    result["failed_12"] = failed_12
    if failed_early > 0:
        result["failed_early"] = failed_early
    return result


def draw_chart(runs: list[tuple[str, dict]], output_path: str | None, normalize: bool):
    """Draw a grouped stacked bar chart."""
    labels = [r[0] for r in runs]
    counts_list = [r[1] for r in runs]

    # Categories for stacking (bottom to top), filtering out all-zero categories
    success_keys = [f"success_{t}" for t in TURN_BUCKETS if any(c.get(f"success_{t}", 0) > 0 for c in counts_list)]
    has_other_success = any("success_other" in c for c in counts_list)
    has_failed_12 = any(c.get("failed_12", 0) > 0 for c in counts_list)
    has_failed_early = any("failed_early" in c for c in counts_list)
    categories = (
        success_keys
        + (["success_other"] if has_other_success else [])
        + (["failed_12"] if has_failed_12 else [])
        + (["failed_early"] if has_failed_early else [])
    )
    active_turns = [t for t in TURN_BUCKETS if any(c.get(f"success_{t}", 0) > 0 for c in counts_list)]
    display_names = [f"Pass @ {t} turns" for t in active_turns]
    if has_other_success:
        display_names.append("Pass @ other")
    if has_failed_12:
        display_names.append("Failed (12 turns)")
    if has_failed_early:
        display_names.append("Failed (< 12 turns)")

    # Colors: greens for success (light to dark), reds for failed
    n_success = len(success_keys) + (1 if has_other_success else 0)
    success_colors = plt.cm.Greens(np.linspace(0.25, 0.85, n_success))
    colors = list(success_colors) + ["#d9534f"] + (["#f0a08a"] if has_failed_early else [])

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.5), 6))

    x = np.arange(len(labels))
    width = 0.6

    for i, counts in enumerate(counts_list):
        total = sum(counts.get(k, 0) for k in categories)
        bottom = 0
        for j, cat in enumerate(categories):
            val = counts.get(cat, 0)
            if normalize and total > 0:
                val = val / total * 100
            ax.bar(
                x[i],
                val,
                width,
                bottom=bottom,
                color=colors[j],
                edgecolor="white",
                linewidth=0.5,
                label=display_names[j] if i == 0 else None,
            )
            # Add count text if segment is large enough
            threshold = 3 if normalize else total * 0.03
            if val > threshold:
                if normalize and total > 0:
                    pct = counts.get(cat, 0) / total * 100
                    text = f"{pct:.1f}%"
                else:
                    text = f"{int(counts.get(cat, 0))}"
                ax.text(x[i], bottom + val / 2, text, ha="center", va="center", fontsize=7, fontweight="bold")
            bottom += val

    # Label x-ticks with run name and total repetition count
    tick_labels = []
    for i, counts in enumerate(counts_list):
        total = sum(counts.get(k, 0) for k in categories)
        tick_labels.append(f"{labels[i]}\n(n={total})")
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% of repetitions" if normalize else "Number of repetitions")
    ax.set_title("Turn Distribution: Success vs Failure by Turn Count")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved chart to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare turn distributions across multiple runs.")
    parser.add_argument(
        "--run",
        nargs=2,
        action="append",
        metavar=("LABEL", "PATH"),
        required=True,
        help='A run to include: --run "label" path/to/result.json',
    )
    parser.add_argument("-o", "--output", help="Output image path (e.g., chart.png)")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize bars to 100%% (useful when runs have different repetition counts)",
    )
    args = parser.parse_args()

    runs = []
    for label, path in args.run:
        data = load_result(path)
        counts = count_by_turns(data)
        runs.append((label, counts))
        total = sum(counts.values())
        print(f"{label}: {counts}  (total={total})")

    draw_chart(runs, args.output, args.normalize)


if __name__ == "__main__":
    main()
