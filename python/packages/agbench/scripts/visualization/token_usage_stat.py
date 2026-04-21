import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FIGURE_TITLE_KEYWORD = "Think Mode"
FIGURE_FILE_SUFFIX = "_think"

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# dirs = {
#     "Qwen3-0.6B": os.path.join(BASE_DIR, "output_0.6b_nothink"),
#     "Qwen3-1.7B": os.path.join(BASE_DIR, "output_1.7b_nothink"),
#     "Qwen3-4B": os.path.join(BASE_DIR, "output_4b_nothink"),
#     "Qwen3-8B": os.path.join(BASE_DIR, "output_8b_nothink"),
# }
dirs = {
    "Qwen3-0.6B": os.path.join(BASE_DIR, "output_0.6b_think"),
    "Qwen3-1.7B": os.path.join(BASE_DIR, "output_1.7b_think"),
    "Qwen3-4B": os.path.join(BASE_DIR, "output_4b_think"),
    "Qwen3-8B": os.path.join(BASE_DIR, "output_8b_think"),
}


def parse_token_usage_from_console_log(log_path: str) -> list[dict]:
    """Parse a console_log.txt file and extract token usage and latency from each LLM call.

    Returns a list of dicts, one per LLM call (turn), each with keys:
        prompt_tokens, completion_tokens, total_tokens, reasoning_tokens, latency_s
    """
    with open(log_path, "r") as f:
        content = f.read()

    usage_pattern = r"CompletionUsage\(completion_tokens=(\d+),\s*prompt_tokens=(\d+),\s*total_tokens=(\d+),\s*completion_tokens_details=\w+,\s*prompt_tokens_details=\w+,\s*reasoning_tokens=(\d+)\)"
    latency_pattern = r"\[runtime\] name=coder seconds=([\d.]+)"

    usage_matches = re.findall(usage_pattern, content)
    latency_matches = re.findall(latency_pattern, content)

    results = []
    for i, m in enumerate(usage_matches):
        entry = {
            "completion_tokens": int(m[0]),
            "prompt_tokens": int(m[1]),
            "total_tokens": int(m[2]),
            "reasoning_tokens": int(m[3]),
            "latency_s": float(latency_matches[i]) if i < len(latency_matches) else None,
        }
        results.append(entry)
    return results


def collect_token_data(output_dir: str) -> pd.DataFrame:
    """Collect token usage data from all trials in an output directory.

    Returns a DataFrame with columns:
        task, trial, turn, prompt_tokens, completion_tokens, total_tokens, reasoning_tokens
    """
    results_dir = os.path.join(output_dir, "Results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"No Results directory in {output_dir}")

    scenario_runs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    if not scenario_runs:
        raise FileNotFoundError(f"No scenario run directories in {results_dir}")

    rows = []
    for scenario_run in scenario_runs:
        scenario_run_path = os.path.join(results_dir, scenario_run)
        scenario_folders = [
            d for d in os.listdir(scenario_run_path) if os.path.isdir(os.path.join(scenario_run_path, d))
        ]
        for scenario_folder in scenario_folders:
            scenario_path = os.path.join(scenario_run_path, scenario_folder)
            for task_name in sorted(os.listdir(scenario_path)):
                task_path = os.path.join(scenario_path, task_name)
                if not os.path.isdir(task_path):
                    continue
                for trial_name in sorted(os.listdir(task_path)):
                    trial_path = os.path.join(task_path, trial_name)
                    log_path = os.path.join(trial_path, "console_log.txt")
                    if not os.path.isfile(log_path):
                        continue
                    usages = parse_token_usage_from_console_log(log_path)
                    for turn_idx, usage in enumerate(usages):
                        rows.append(
                            {
                                "task": task_name,
                                "trial": int(trial_name),
                                "turn": turn_idx + 1,
                                **usage,
                            }
                        )

    return pd.DataFrame(rows)


def analyze_token_distributions(output_dirs: dict[str, str]) -> pd.DataFrame:
    """Analyze token distributions across multiple output directories.

    Args:
        output_dirs: dict mapping label -> output directory path

    Returns:
        Combined DataFrame with a 'model' column for the label.
    """
    all_dfs = []
    for label, path in output_dirs.items():
        print(f"Processing {label} from {path}...")
        df = collect_token_data(path)
        df["model"] = label
        all_dfs.append(df)
        print(f"  Found {len(df)} LLM calls across {df['task'].nunique()} tasks")

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def main():

    df = analyze_token_distributions(dirs)
    print(f"\nTotal records: {len(df)}")
    print(df.head())

    # --- P50/P99 input/output tokens and latency per turn by model ---
    stats_per_turn = (
        df.groupby(["model", "turn"])
        .agg(
            p50_prompt_tokens=("prompt_tokens", lambda x: x.quantile(0.5)),
            p99_prompt_tokens=("prompt_tokens", lambda x: x.quantile(0.99)),
            p50_completion_tokens=("completion_tokens", lambda x: x.quantile(0.5)),
            p99_completion_tokens=("completion_tokens", lambda x: x.quantile(0.99)),
            p50_total_tokens=("total_tokens", lambda x: x.quantile(0.5)),
            p99_total_tokens=("total_tokens", lambda x: x.quantile(0.99)),
            p50_latency_s=("latency_s", lambda x: x.quantile(0.5)),
            p99_latency_s=("latency_s", lambda x: x.quantile(0.99)),
            count=("prompt_tokens", "count"),
        )
        .round(2)
    )
    print("\nP50/P99 Input/Output Tokens & Latency Per Turn (by model):")
    print("=" * 80)
    print(stats_per_turn)

    # --- Overall P50/P99 per model ---
    overall_stats = (
        df.groupby("model")
        .agg(
            p50_prompt_tokens=("prompt_tokens", lambda x: x.quantile(0.5)),
            p99_prompt_tokens=("prompt_tokens", lambda x: x.quantile(0.99)),
            p50_completion_tokens=("completion_tokens", lambda x: x.quantile(0.5)),
            p99_completion_tokens=("completion_tokens", lambda x: x.quantile(0.99)),
            p50_total_tokens=("total_tokens", lambda x: x.quantile(0.5)),
            p99_total_tokens=("total_tokens", lambda x: x.quantile(0.99)),
            p50_latency_s=("latency_s", lambda x: x.quantile(0.5)),
            p99_latency_s=("latency_s", lambda x: x.quantile(0.99)),
            total_turns=("prompt_tokens", "count"),
            num_tasks=("task", "nunique"),
        )
        .round(2)
    )
    print("\nOverall P50/P99 Input/Output Tokens & Latency Per Model:")
    print("=" * 80)
    print(overall_stats)

    model_order = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    markers = ["o", "s", "D", "^"]

    def plot_tokens_per_turn(pct_key, pct_label, filename):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for model, color, marker in zip(model_order, colors, markers):
            if model not in stats_per_turn.index:
                continue
            model_data = stats_per_turn.loc[model]
            turns = model_data.index.get_level_values("turn")
            for ax, col in zip(axes, [f"{pct_key}_prompt_tokens", f"{pct_key}_completion_tokens"]):
                ax.plot(turns, model_data[col], color=color, marker=marker, label=model, linewidth=2, markersize=8)
                for t, v in zip(turns, model_data[col]):
                    ax.annotate(
                        f"{v:.0f}",
                        (t, v),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=7,
                        color=color,
                    )
        axes[0].set_xlabel("Turn Number")
        axes[0].set_ylabel("Prompt Tokens (Input)")
        axes[0].set_title(f"{pct_label} Input Tokens per Turn")
        axes[0].legend()
        axes[0].set_xticks(range(1, 7))
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("Turn Number")
        axes[1].set_ylabel("Completion Tokens (Output)")
        axes[1].set_title(f"{pct_label} Output Tokens per Turn")
        axes[1].legend()
        axes[1].set_xticks(range(1, 7))
        axes[1].grid(True, alpha=0.3)
        plt.suptitle(f"{pct_label} Tokens per Turn Across Qwen3 Model Sizes ({FIGURE_TITLE_KEYWORD})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "reports", filename), dpi=150, bbox_inches="tight")
        plt.close()

    plot_tokens_per_turn("p50", "P50", f"tokens_per_turn_p50{FIGURE_FILE_SUFFIX}.png")
    plot_tokens_per_turn("p99", "P99", f"tokens_per_turn_p99{FIGURE_FILE_SUFFIX}.png")

    def plot_latency_per_turn(pct_key, pct_label, filename):
        fig, ax = plt.subplots(figsize=(10, 6))
        for model, color, marker in zip(model_order, colors, markers):
            if model not in stats_per_turn.index:
                continue
            model_data = stats_per_turn.loc[model]
            turns = model_data.index.get_level_values("turn")
            ax.plot(
                turns,
                model_data[f"{pct_key}_latency_s"],
                color=color,
                marker=marker,
                label=model,
                linewidth=2,
                markersize=8,
            )
            for t, v in zip(turns, model_data[f"{pct_key}_latency_s"]):
                ax.annotate(
                    f"{v:.2f}", (t, v), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=7, color=color
                )
        ax.set_xlabel("Turn Number")
        ax.set_ylabel("Latency (s)")
        ax.set_title(f"{pct_label} E2E LLM Latency per Turn")
        ax.legend()
        ax.set_xticks(range(1, 7))
        ax.grid(True, alpha=0.3)
        plt.suptitle(f"E2E LLM Latency {pct_label} Across Qwen3 Model Sizes ({FIGURE_TITLE_KEYWORD})", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE_DIR, "reports", filename), dpi=150, bbox_inches="tight")
        plt.close()

    plot_latency_per_turn("p50", "P50", f"latency_per_turn_p50{FIGURE_FILE_SUFFIX}.png")
    plot_latency_per_turn("p99", "P99", f"latency_per_turn_p99{FIGURE_FILE_SUFFIX}.png")


def parse_step_latencies(log_path: str) -> list[tuple[str, float]]:
    """Parse a console_log.txt and extract (step_name, latency_s) pairs in order."""
    runtime_pattern = re.compile(r"\[runtime\]\s+name=(\S+)\s+seconds=([\d.]+)")
    steps = []
    with open(log_path, "r") as f:
        for line in f:
            m = runtime_pattern.search(line)
            if m:
                steps.append((m.group(1), float(m.group(2))))
    return steps


def collect_step_latencies(output_dir: str) -> pd.DataFrame:
    """Collect per-step latency data from all trials in an output directory.

    Returns a DataFrame with columns: task, trial, step_index, step_name, latency_s
    """
    results_dir = os.path.join(output_dir, "Results")
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"No Results directory in {output_dir}")

    rows = []
    scenario_runs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    for scenario_run in scenario_runs:
        scenario_run_path = os.path.join(results_dir, scenario_run)
        scenario_folders = [
            d for d in os.listdir(scenario_run_path) if os.path.isdir(os.path.join(scenario_run_path, d))
        ]
        for scenario_folder in scenario_folders:
            scenario_path = os.path.join(scenario_run_path, scenario_folder)
            for task_name in sorted(os.listdir(scenario_path)):
                task_path = os.path.join(scenario_path, task_name)
                if not os.path.isdir(task_path):
                    continue
                for trial_name in sorted(os.listdir(task_path)):
                    trial_path = os.path.join(task_path, trial_name)
                    log_path = os.path.join(trial_path, "console_log.txt")
                    if not os.path.isfile(log_path):
                        continue
                    steps = parse_step_latencies(log_path)
                    for step_idx, (step_name, latency) in enumerate(steps):
                        rows.append({
                            "task": task_name,
                            "trial": trial_name,
                            "step_index": step_idx,
                            "step_name": step_name,
                            "latency_s": latency,
                        })

    return pd.DataFrame(rows)


def plot_stacked_latency_comparison():
    """Plot a stacked bar chart comparing per-step latency across setups.

    Each sequential step (step_index) gets its own color and stacked segment.
    For single-model setups: Turn 1 Coder, Turn 1 Executor, Turn 2 Coder, ...
    For two-turn setup: coder_small, executor_1, coder_large, executor_2
    Annotations show mean latency (over tasks that have that step) and % of tasks.
    """

    setup_dirs = {
        "Two-Turn\n(0.6B-NT → 4B-T)": os.path.join(BASE_DIR, "outputs", "two-turn-0.6nt-4t"),
        "Two-Turn\n(1.7B-NT → 4B-T)": os.path.join(BASE_DIR, "outputs", "two-turn-1.7nt-4t"),
    }

    # For each setup, compute per-step_index stats
    # step_index is the sequential position in the log (0, 1, 2, 3, ...)
    setup_data = {}  # label -> list of {step_index, step_label, mean, pct, count}
    setup_e2e = {}   # label -> {"mean": float, "median": float, "p90": float, "std": float}
    max_steps = 0
    for label, path in setup_dirs.items():
        print(f"Processing {label} from {path}...")
        df = collect_step_latencies(path)
        if df.empty:
            print(f"  No data found, skipping.")
            continue

        total_tasks = df[["task", "trial"]].drop_duplicates().shape[0]

        # Compute e2e latency: sum all steps per (task, trial)
        e2e = df.groupby(["task", "trial"])["latency_s"].sum()
        setup_e2e[label] = {
            "mean": e2e.mean(),
            "median": e2e.median(),
            "p90": e2e.quantile(0.9),
            "std": e2e.std(),
        }
        print(f"  E2E latency: mean={e2e.mean():.2f}s, median={e2e.median():.2f}s, "
              f"p90={e2e.quantile(0.9):.2f}s, std={e2e.std():.2f}s")

        # For each step_index, compute stats
        step_list = []
        for step_idx, group in df.groupby("step_index"):
            n_tasks = group[["task", "trial"]].drop_duplicates().shape[0]
            mean_lat = group["latency_s"].mean()
            pct = 100.0 * n_tasks / total_tasks
            # Build a human-readable label from the step_name
            # For single-model: step 0=coder, 1=executor, 2=coder, 3=executor...
            # For two-turn: step 0=coder_small, 1=executor_1, 2=coder_large, 3=executor_2
            step_names = group["step_name"].unique()
            step_name = step_names[0] if len(step_names) == 1 else "/".join(step_names)
            # Create display label
            if step_name in ("coder", "executor"):
                turn_num = step_idx // 2 + 1
                role = "Coder" if step_name == "coder" else "Executor"
                display = f"T{turn_num} {role}"
            else:
                name_map = {
                    "coder_small": "T1 Coder (0.6B)",
                    "executor_1": "T1 Executor",
                    "coder_large": "T2 Coder (4B)",
                    "executor_2": "T2 Executor",
                }
                display = name_map.get(step_name, step_name)

            step_list.append({
                "step_index": step_idx,
                "step_label": display,
                "step_name": step_name,
                "mean": mean_lat,
                "pct": pct,
                "count": n_tasks,
                "total": total_tasks,
            })

        setup_data[label] = step_list
        max_steps = max(max_steps, len(step_list))
        print(f"  Total tasks: {total_tasks}, steps: {len(step_list)}")
        for s in step_list:
            print(f"    [{s['step_index']}] {s['step_label']}: mean={s['mean']:.2f}s, "
                  f"{s['pct']:.1f}% ({s['count']}/{s['total']})")

    # Color palette: alternate coder/executor shades per turn
    # Use a colormap with enough distinct colors
    coder_colors = ["#2196F3", "#9C27B0", "#009688", "#3F51B5", "#00BCD4", "#673AB7"]
    executor_colors = ["#FF9800", "#FFC107", "#FF5722", "#FFEB3B", "#E91E63", "#CDDC39"]

    def get_step_color(step_index, step_name):
        """Assign distinct color based on step index and role."""
        if "executor" in step_name:
            return executor_colors[min(step_index // 2, len(executor_colors) - 1)]
        else:
            return coder_colors[min(step_index // 2, len(coder_colors) - 1)]

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(setup_data))
    bar_width = 0.5
    setup_labels = list(setup_data.keys())

    # Track legend entries
    legend_entries = {}  # label -> handle

    for i, label in enumerate(setup_labels):
        steps = setup_data[label]
        bottom = 0.0
        for s in steps:
            val = s["mean"]
            pct = s["pct"]
            step_label = s["step_label"]
            color = get_step_color(s["step_index"], s["step_name"])

            bar = ax.bar(i, val, bar_width, bottom=bottom, color=color,
                         edgecolor="black", linewidth=0.6)

            if step_label not in legend_entries:
                legend_entries[step_label] = bar

            # Annotate
            mid = bottom + val / 2
            if val > 5.0:
                ax.text(i, mid, f"{val:.1f}s\n({pct:.0f}%)",
                        ha="center", va="center", fontsize=8, fontweight="bold", color="white")
            elif val > 1.5:
                ax.text(i, mid, f"{val:.1f}s ({pct:.0f}%)",
                        ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            elif val > 0.2:
                offset_x = bar_width / 2 + 0.05
                ax.annotate(f"{step_label}: {val:.1f}s ({pct:.0f}%)",
                            xy=(i + bar_width / 2, mid),
                            xytext=(i + offset_x + 0.15, mid),
                            fontsize=6.5,
                            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
                            va="center", ha="left")
            bottom += val
        # E2E stats on top
        e2e = setup_e2e.get(label, {})
        e2e_mean = e2e.get("mean", bottom)
        e2e_median = e2e.get("median", 0)
        e2e_p90 = e2e.get("p90", 0)
        ax.text(i, bottom + 0.5,
                f"Avg E2E: {e2e_mean:.1f}s\nMedian: {e2e_median:.1f}s\nP90: {e2e_p90:.1f}s",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(setup_labels, fontsize=11)
    ax.set_ylabel("Mean Latency per Task (seconds)", fontsize=12)
    ax.set_title("Per-Turn Latency Breakdown Across Setups (HumanEval)", fontsize=14)
    ax.grid(axis="y", alpha=0.3)

    # Legend from all unique step labels
    ax.legend(
        [legend_entries[k] for k in legend_entries],
        list(legend_entries.keys()),
        fontsize=8, loc="upper left", ncol=2,
    )

    plt.tight_layout()
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "reports", "stacked_latency_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved stacked latency comparison to {out_path}")


if __name__ == "__main__":
    main()
    plot_stacked_latency_comparison()
