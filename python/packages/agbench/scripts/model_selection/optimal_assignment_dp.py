"""DP-based optimal model assignment for HumanEval tasks under a latency budget.

Parses console logs to extract first-turn pass rates and first-turn coder
latencies, then solves a multiple-choice knapsack problem via dynamic
programming to maximize total (expected) pass rate subject to a total latency
budget.

Usage:
    cd python/packages/agbench/outputs
    python ../scripts/optimal_assignment_dp.py --budget 500

    # Use overall pass rates from CSV instead of first-turn:
    python ../scripts/optimal_assignment_dp.py --budget 500 --use-overall

    # Use per-model average latency instead of per-task-model:
    python ../scripts/optimal_assignment_dp.py --budget 500 --avg-latency
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODELS = [
    "0.6b_nothink", "1.7b_nothink", "4b_nothink", "8b_nothink",
    "0.6b_think", "1.7b_think", "4b_think", "8b_think",
]

OUTPUT_DIRS = {m: f"output_{m}" for m in MODELS}

PASS_RATES_CSV = "../reports/task_pass_rates.csv"

# Latency discretization resolution (milliseconds)
RESOLUTION_MS = 1  # 1 ms granularity


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_first_turn(log_path: Path):
    """Parse a single console_log.txt and return (first_turn_passed, first_turn_coder_latency_s).

    First-turn passes if the task needed only 1 coder invocation (i.e., the
    scenario would have succeeded with max_turns=2: 1 coder + 1 executor).
    We detect this by counting [runtime] name=coder entries: if there is
    exactly 1, the first (and only) coder attempt passed.

    Returns (None, None) if the log cannot be parsed.
    """
    text = log_path.read_text(errors="replace")

    # Find all coder runtime entries
    coder_latencies = re.findall(r"\[runtime\]\s*name=coder\s+seconds=([\d.]+)", text)
    if not coder_latencies:
        return None, None

    first_latency = float(coder_latencies[0])

    # First-turn passes iff there was exactly 1 coder invocation
    # (the test passed on the first attempt, so no retry was needed).
    n_coder_turns = len(coder_latencies)
    first_passed = (n_coder_turns == 1)

    return first_passed, first_latency


def collect_first_turn_data(output_dirs: dict):
    """Collect first-turn data from all model output directories.

    Returns:
        tasks: sorted list of task names
        first_turn_pass: dict[(task, model)] -> list of bool (one per trial)
        first_turn_latency: dict[(task, model)] -> list of float seconds
    """
    first_turn_pass = defaultdict(list)
    first_turn_latency = defaultdict(list)
    all_tasks = set()

    for model, dirname in output_dirs.items():
        base = Path(dirname) / "Results" / "human_eval"
        # Find the AgentChat subdirectory
        agent_dirs = list(base.glob("human_eval_AgentChat*"))
        if not agent_dirs:
            print(f"Warning: no AgentChat dir in {base}, skipping {model}", file=sys.stderr)
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
                passed, latency = parse_first_turn(log_path)
                if passed is not None:
                    first_turn_pass[(task, model)].append(passed)
                if latency is not None:
                    first_turn_latency[(task, model)].append(latency)

    tasks = sorted(all_tasks, key=lambda t: int(t.replace("HumanEval_", "")))
    return tasks, first_turn_pass, first_turn_latency


def load_overall_pass_rates(csv_path: str):
    """Load overall pass rates from the CSV (as produced by task_pass_rates.py)."""
    tasks = []
    rates = {}  # (task, model) -> float
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row["Task"]
            tasks.append(task)
            for model in MODELS:
                val = row.get(model, "")
                rates[(task, model)] = float(val) if val else 0.0
    return tasks, rates


# ---------------------------------------------------------------------------
# DP solver
# ---------------------------------------------------------------------------

def solve_knapsack(pass_rates, latencies, budget_ms, resolution_ms=RESOLUTION_MS):
    """Multiple-choice knapsack via DP.

    Args:
        pass_rates: np.array (n_tasks, n_models) — expected pass rate
        latencies: np.array (n_tasks, n_models) — latency in seconds
        budget_ms: int — total latency budget in milliseconds
        resolution_ms: int — discretization resolution

    Returns:
        assignment: list of model indices (one per task)
        total_pass_rate: float
        total_latency_ms: int
    """
    n_tasks, n_models = pass_rates.shape

    # Discretize latencies to integer units
    lat_int = np.round(latencies * 1000 / resolution_ms).astype(int)  # seconds → ms → units
    budget_units = int(np.round(budget_ms / resolution_ms))

    # DP table: dp[b] = best cumulative pass rate achievable with budget b
    # We use rolling 1D DP (only keep current and previous task)
    dp_prev = np.full(budget_units + 1, -np.inf)
    dp_prev[0] = 0.0

    # Choice backtracking: choice[t][b] = model index chosen for task t at budget b
    choice = np.zeros((n_tasks, budget_units + 1), dtype=np.int32)

    for t in range(n_tasks):
        dp_curr = np.full(budget_units + 1, -np.inf)
        for m in range(n_models):
            cost = lat_int[t, m]
            value = pass_rates[t, m]
            if cost > budget_units:
                continue
            # For all budgets b where we can afford cost for model m
            for b in range(cost, budget_units + 1):
                candidate = dp_prev[b - cost] + value
                if candidate > dp_curr[b]:
                    dp_curr[b] = candidate
                    choice[t][b] = m
        dp_prev = dp_curr

    # Find best budget level
    best_b = int(np.argmax(dp_prev))
    best_val = dp_prev[best_b]

    # Backtrack
    assignment = [0] * n_tasks
    b = best_b
    for t in range(n_tasks - 1, -1, -1):
        m = choice[t][b]
        assignment[t] = m
        b -= lat_int[t, m]

    total_latency_ms = best_b * resolution_ms
    return assignment, float(best_val), total_latency_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Optimal model assignment via DP knapsack")
    parser.add_argument("--budget", type=float, required=True,
                        help="Total latency budget in seconds")
    parser.add_argument("--use-overall", action="store_true",
                        help="Use overall pass rates from CSV instead of first-turn pass rates")
    parser.add_argument("--avg-latency", action="store_true",
                        help="Use per-model average latency instead of per-task-model latency")
    parser.add_argument("--resolution", type=int, default=RESOLUTION_MS,
                        help="Latency discretization resolution in ms (default: 1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for assignment results")
    args = parser.parse_args()

    budget_ms = int(args.budget * 1000)

    # -- Collect data --
    print("Collecting first-turn data from logs...", file=sys.stderr)
    tasks, ft_pass, ft_latency = collect_first_turn_data(OUTPUT_DIRS)
    n_tasks = len(tasks)
    n_models = len(MODELS)
    print(f"Found {n_tasks} tasks, {n_models} models", file=sys.stderr)

    # -- Build pass rate matrix --
    if args.use_overall:
        print("Using overall pass rates from CSV", file=sys.stderr)
        _, overall_rates = load_overall_pass_rates(PASS_RATES_CSV)
        pass_rates = np.zeros((n_tasks, n_models))
        for i, task in enumerate(tasks):
            for j, model in enumerate(MODELS):
                pass_rates[i, j] = overall_rates.get((task, model), 0.0)
    else:
        print("Using first-turn pass rates from logs", file=sys.stderr)
        pass_rates = np.zeros((n_tasks, n_models))
        for i, task in enumerate(tasks):
            for j, model in enumerate(MODELS):
                trials = ft_pass.get((task, model), [])
                if trials:
                    pass_rates[i, j] = sum(trials) / len(trials)
                else:
                    pass_rates[i, j] = 0.0

    # -- Build latency matrix --
    latencies = np.zeros((n_tasks, n_models))
    for i, task in enumerate(tasks):
        for j, model in enumerate(MODELS):
            lats = ft_latency.get((task, model), [])
            if lats:
                latencies[i, j] = np.mean(lats)
            else:
                latencies[i, j] = 1e6  # very large = effectively infeasible

    if args.avg_latency:
        # Collapse to per-model average latency
        model_avg = np.zeros(n_models)
        for j, model in enumerate(MODELS):
            all_lats = []
            for i, task in enumerate(tasks):
                lats = ft_latency.get((task, model), [])
                all_lats.extend(lats)
            model_avg[j] = np.mean(all_lats) if all_lats else 1e6
        print(f"Per-model avg latency (s): {dict(zip(MODELS, model_avg))}", file=sys.stderr)
        latencies = np.tile(model_avg, (n_tasks, 1))

    # -- Print summary stats --
    print(f"\nLatency budget: {args.budget:.1f}s ({budget_ms} ms)", file=sys.stderr)
    print(f"Resolution: {args.resolution} ms", file=sys.stderr)

    for j, model in enumerate(MODELS):
        avg_pass = np.mean(pass_rates[:, j])
        avg_lat = np.mean(latencies[:, j])
        total_lat = np.sum(latencies[:, j])
        print(f"  {model:16s}  avg_pass={avg_pass:.3f}  avg_lat={avg_lat:.3f}s  total_lat={total_lat:.1f}s",
              file=sys.stderr)

    # -- Cheapest model baseline --
    cheapest_model_idx = np.argmin(np.sum(latencies, axis=0))
    cheapest_total_lat = np.sum(latencies[:, cheapest_model_idx])
    cheapest_total_pass = np.sum(pass_rates[:, cheapest_model_idx])
    print(f"\nCheapest model baseline: {MODELS[cheapest_model_idx]} "
          f"(total_lat={cheapest_total_lat:.1f}s, total_pass={cheapest_total_pass:.1f}/{n_tasks})",
          file=sys.stderr)

    # -- Best model baseline --
    best_model_idx = np.argmax(np.mean(pass_rates, axis=0))
    best_total_lat = np.sum(latencies[:, best_model_idx])
    best_total_pass = np.sum(pass_rates[:, best_model_idx])
    print(f"Best model baseline:     {MODELS[best_model_idx]} "
          f"(total_lat={best_total_lat:.1f}s, total_pass={best_total_pass:.1f}/{n_tasks})",
          file=sys.stderr)

    # -- Solve DP --
    print(f"\nSolving DP...", file=sys.stderr)
    assignment, total_pass, total_lat_ms = solve_knapsack(
        pass_rates, latencies, budget_ms, args.resolution
    )
    total_lat_s = total_lat_ms / 1000.0

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"DP Result:", file=sys.stderr)
    print(f"  Total expected pass rate: {total_pass:.1f}/{n_tasks} "
          f"({total_pass/n_tasks*100:.1f}%)", file=sys.stderr)
    print(f"  Total latency: {total_lat_s:.1f}s (budget: {args.budget:.1f}s)", file=sys.stderr)

    # Model usage distribution
    model_counts = defaultdict(int)
    for m_idx in assignment:
        model_counts[MODELS[m_idx]] += 1
    print(f"  Model distribution:", file=sys.stderr)
    for model in MODELS:
        if model_counts[model] > 0:
            print(f"    {model:16s}: {model_counts[model]:3d} tasks", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    # -- Output --
    output_path = args.output or "../reports/optimal_assignment.csv"
    with open(output_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Task", "Assigned_Model", "Pass_Rate", "Latency_s"])
        for i, task in enumerate(tasks):
            m_idx = assignment[i]
            w.writerow([
                task,
                MODELS[m_idx],
                f"{pass_rates[i, m_idx]:.3f}",
                f"{latencies[i, m_idx]:.3f}",
            ])
    print(f"\nAssignment saved to {output_path}", file=sys.stderr)

    # -- Also print Pareto sweep --
    print(f"\n--- Pareto sweep ---", file=sys.stderr)
    # Try a range of budgets from cheapest to most expensive
    min_budget_s = cheapest_total_lat
    max_budget_s = best_total_lat
    steps = 10
    print(f"{'Budget(s)':>10s} {'PassRate':>10s} {'Pct':>6s} {'Latency(s)':>10s}", file=sys.stderr)
    for step in range(steps + 1):
        b = min_budget_s + (max_budget_s - min_budget_s) * step / steps
        b_ms = int(b * 1000)
        a, pr, lat = solve_knapsack(pass_rates, latencies, b_ms, args.resolution)
        print(f"{b:10.1f} {pr:10.1f} {pr/n_tasks*100:5.1f}% {lat/1000:10.1f}", file=sys.stderr)


if __name__ == "__main__":
    main()
