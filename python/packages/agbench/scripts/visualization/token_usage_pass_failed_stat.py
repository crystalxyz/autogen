#!/usr/bin/env python3
"""Plot token usage per turn for passed vs. failed tasks side by side.

Aggregates data from ALL output_* directories under the project root.
Produces a 2×2 grid: rows = P50 / P99, columns = Passed / Failed.
Only turns 1-3 are shown; per-turn sample counts are shown in the x-tick labels.

Usage:
    python scripts/token_usage_pass_failed_stat.py [-o output.png]

Example:
    python scripts/token_usage_pass_failed_stat.py
    python scripts/token_usage_pass_failed_stat.py -o reports/token_pass_fail.png
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_token_usage_from_console_log(log_path: str) -> list[dict]:
    """Extract token usage from a console_log.txt file.

    Returns a list of dicts (one per LLM call / turn) with keys:
        prompt_tokens, completion_tokens, total_tokens, reasoning_tokens
    """
    with open(log_path, "r") as f:
        content = f.read()

    usage_pattern = (
        r"CompletionUsage\("
        r"completion_tokens=(\d+),\s*"
        r"prompt_tokens=(\d+),\s*"
        r"total_tokens=(\d+),\s*"
        r"completion_tokens_details=\w+,\s*"
        r"prompt_tokens_details=\w+,\s*"
        r"reasoning_tokens=(\d+)\)"
    )
    matches = re.findall(usage_pattern, content)

    results = []
    for m in matches:
        results.append(
            {
                "completion_tokens": int(m[0]),
                "prompt_tokens": int(m[1]),
                "total_tokens": int(m[2]),
                "reasoning_tokens": int(m[3]),
            }
        )
    return results


def load_result_json(results_dir: str) -> dict:
    """Load result.json from the first scenario-run directory found.

    Returns a dict mapping task_id -> {trial_idx (str) -> success (bool)}.
    """
    result_json_path = None
    for entry in os.listdir(results_dir):
        candidate = os.path.join(results_dir, entry, "result.json")
        if os.path.isfile(candidate):
            result_json_path = candidate
            break

    if result_json_path is None:
        raise FileNotFoundError(f"No result.json found under {results_dir}")

    with open(result_json_path) as f:
        data = json.load(f)

    task_success: dict[str, dict[str, bool]] = {}
    for task_id, task_info in data.get("tasks", {}).items():
        task_success[task_id] = {
            rep_idx: rep["success"]
            for rep_idx, rep in task_info.get("repetitions", {}).items()
        }
    return task_success


def collect_token_data_with_outcome(output_dir: str) -> pd.DataFrame:
    """Collect per-turn token usage tagged with pass/fail outcome.

    Returns a DataFrame with columns:
        task, trial, turn, prompt_tokens, completion_tokens, total_tokens,
        reasoning_tokens, success
    """
    results_dir = os.path.join(output_dir, "Results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"No Results/ directory in {output_dir}")

    # Load pass/fail from result.json
    task_success = load_result_json(results_dir)

    rows = []
    for scenario_run in sorted(os.listdir(results_dir)):
        scenario_run_path = os.path.join(results_dir, scenario_run)
        if not os.path.isdir(scenario_run_path):
            continue

        for scenario_folder in sorted(os.listdir(scenario_run_path)):
            scenario_folder_path = os.path.join(scenario_run_path, scenario_folder)
            if not os.path.isdir(scenario_folder_path):
                continue

            for task_name in sorted(os.listdir(scenario_folder_path)):
                task_path = os.path.join(scenario_folder_path, task_name)
                if not os.path.isdir(task_path):
                    continue

                for trial_name in sorted(os.listdir(task_path)):
                    trial_path = os.path.join(task_path, trial_name)
                    if not os.path.isdir(trial_path):
                        continue

                    log_path = os.path.join(trial_path, "console_log.txt")
                    if not os.path.isfile(log_path):
                        continue

                    # Determine success for this (task, trial)
                    success = task_success.get(task_name, {}).get(trial_name, None)
                    if success is None:
                        continue  # skip if not found in result.json

                    usages = parse_token_usage_from_console_log(log_path)
                    for turn_idx, usage in enumerate(usages):
                        rows.append(
                            {
                                "task": task_name,
                                "trial": trial_name,
                                "turn": turn_idx + 1,
                                "success": success,
                                **usage,
                            }
                        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SHOW_TURNS = [1, 2, 3]

# 2 bars per turn group: input | output
BAR_WIDTH = 0.3
BAR_SPECS = [
    ("prompt_col",     "Input Tokens",  "#1f77b4"),
    ("completion_col", "Output Tokens", "#ff7f0e"),
]


def stats_per_turn(subset: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Compute input/output token percentile per turn, filtered to SHOW_TURNS.

    Only includes turns that actually have data; missing turns are omitted.
    """
    if subset.empty:
        return pd.DataFrame(columns=["turn", "prompt_col", "completion_col", "count"])
    filtered = subset[subset["turn"].isin(SHOW_TURNS)]
    if filtered.empty:
        return pd.DataFrame(columns=["turn", "prompt_col", "completion_col", "count"])
    return (
        filtered.groupby("turn")
        .agg(
            prompt_col=("prompt_tokens", lambda x: x.quantile(pct)),
            completion_col=("completion_tokens", lambda x: x.quantile(pct)),
            count=("prompt_tokens", "count"),
        )
        .reset_index()
        .sort_values("turn")
    )


def draw_pass_fail_row(axes, passed_stats: pd.DataFrame, failed_stats: pd.DataFrame,
                       pct_label: str) -> None:
    """Fill one row (2 axes) with passed/failed bar charts for a given percentile."""
    all_vals = []
    for stats in (passed_stats, failed_stats):
        for col in ("prompt_col", "completion_col"):
            if col in stats.columns:
                all_vals.extend(stats[col].dropna().tolist())
    y_max = max(all_vals) if all_vals else 1

    for ax, stats, outcome in [
        (axes[0], passed_stats, "Passed"),
        (axes[1], failed_stats, "Failed"),
    ]:
        if stats.empty or stats[["prompt_col", "completion_col"]].isna().all().all():
            ax.text(0.5, 0.5, f"No {outcome} tasks", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{pct_label} — {outcome} Tasks")
            continue

        turns = stats["turn"].tolist()
        counts = stats["count"].tolist()
        x = np.arange(len(turns))

        for (col, bar_label, color), offset in zip(BAR_SPECS, [-0.5, 0.5]):
            bars = ax.bar(
                x + offset * BAR_WIDTH,
                stats[col].fillna(0),
                BAR_WIDTH,
                label=bar_label,
                color=color,
                alpha=0.85,
            )
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h + y_max * 0.01,
                        f"{h:.0f}",
                        ha="center", va="bottom", fontsize=7, color=color,
                    )

        tick_labels = [f"Turn {t}\n(n={n})" for t, n in zip(turns, counts)]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Tokens")
        ax.set_title(f"{pct_label} — {outcome} Tasks")
        ax.set_ylim(0, y_max * 1.12)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)


def plot_token_usage_pass_fail(df: pd.DataFrame, output_path: str, run_label: str) -> None:
    """Create a 2×2 grid: rows = P50 / P99, columns = Passed / Failed."""

    passed = df[df["success"] == True]
    failed = df[df["success"] == False]

    p50_passed = stats_per_turn(passed, 0.50)
    p50_failed = stats_per_turn(failed, 0.50)
    p99_passed = stats_per_turn(passed, 0.99)
    p99_failed = stats_per_turn(failed, 0.99)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    draw_pass_fail_row(axes[0], p50_passed, p50_failed, "P50")
    draw_pass_fail_row(axes[1], p99_passed, p99_failed, "P99")

    fig.suptitle(
        f"Token Usage per Turn — Passed vs. Failed Tasks\n{run_label}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=os.path.join(BASE_DIR, "output_8b_think"),
        help="Path to a benchmark output directory (default: output_8b_think)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output PNG path (default: reports/token_usage_pass_failed_stat.png)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(BASE_DIR, args.output_dir)
    output_png = args.output or os.path.join(BASE_DIR, "reports", "token_usage_pass_failed_stat.png")

    run_label = os.path.basename(os.path.normpath(output_dir))

    print(f"Loading data from: {output_dir}")
    df = collect_token_data_with_outcome(output_dir)

    if df.empty:
        print("No data found. Check the output directory.")
        return

    total = len(df)
    passed_rows = (df["success"] == True).sum()
    failed_rows = (df["success"] == False).sum()
    print(f"Total LLM calls: {total}  (passed trials: {passed_rows}, failed trials: {failed_rows})")

    # Summary stats
    for label, subset in [("PASSED", df[df["success"] == True]), ("FAILED", df[df["success"] == False])]:
        if subset.empty:
            print(f"\n{label}: no data")
            continue
        grp = (
            subset[subset["turn"].isin(SHOW_TURNS)]
            .groupby("turn")
            .agg(
                p50_input=("prompt_tokens", lambda x: round(x.quantile(0.50), 1)),
                p99_input=("prompt_tokens", lambda x: round(x.quantile(0.99), 1)),
                p50_output=("completion_tokens", lambda x: round(x.quantile(0.50), 1)),
                p99_output=("completion_tokens", lambda x: round(x.quantile(0.99), 1)),
                count=("prompt_tokens", "count"),
            )
        )
        print(f"\n{label} tasks — p50/p99 tokens per turn (turns 1-3):")
        print(grp.to_string())

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plot_token_usage_pass_fail(df, output_png, run_label)


if __name__ == "__main__":
    main()
