# Overview

## Folder structures

- `scripts/` — Analysis and visualization scripts
- `outputs/` — Raw benchmark run outputs (each subdirectory is one run)
- `reports/csv/` — Derived CSV data files
- `reports/figures/` — Generated plots and visualizations

## Scripts

### Data export / aggregation

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `export_two_turn_csv.py` | Exports all two-turn benchmark results into a single CSV | `outputs/two-turn-*/Results/*/result.json` | `reports/csv/two_turn_results.csv` |
| `two_turn_nc_stats.py` | Computes latency stats, pass/fail ratios (turn1/turn2/fail), and per-turn LLM latency per two-turn-nc setup | `outputs/two-turn-nc-*/Results/*/result.json`, console logs | `reports/csv/two_turn_nc_stats.csv` |
| `task_pass_rates.py` | Per-task pass rate tables across model sizes (overall and first-turn) | `output_*/Results/human_eval/result.json`, console logs | `reports/task_pass_rates.csv`, `reports/task_pass_rates_first_turn.csv` |
| `extract_logprobs_csv.py` | Extracts logprobs statistics and correctness from runs (25+ metrics) | Run dirs with `result.json` and `logprobs_output/` (positional args) | `reports/csv/logprobs.csv`, `reports/csv/logprobs_raw.csv` |
| `recompute_signals_from_raw.py` | Recomputes logprobs signals from raw per-token CSV (change top-k/first-N without re-extraction) | Raw per-token CSV (positional arg) | Summary CSV (via `-o` flag) |

### Model selection / optimization

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `best_model_per_task.py` | Finds cheapest model (lowest first-turn latency) per task at highest first-turn pass rate | `reports/task_pass_rates_first_turn.csv`, console logs | `reports/best_model_per_task.csv` |
| `task_min_model.py` | Finds cheapest model per task with first-turn pass rate > 0.5 threshold | `reports/task_pass_rates_first_turn.csv` | `reports/task_min_model.csv` |
| `optimal_assignment_dp.py` | DP-based optimal model assignment under latency budget (multiple-choice knapsack) | `reports/task_pass_rates.csv`, console logs | Console output |

### Visualization

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `pareto_latency_accuracy.py` | Pareto frontier plots of latency vs accuracy (3 panels: p50, p90, mean) | `reports/csv/two_turn_nc_stats.csv` | `reports/figures/pareto_latency_accuracy.png` |
| `latency_boxplot.py` | Box + strip plot of E2E latency distributions across two-turn setups | `outputs/two-turn-*/Results/*/result.json`, console logs | PNG (hard-coded path) |
| `latency_boxplot_by_outcome.py` | Stacked histograms of E2E latency split by outcome (pass turn 1/2, fail) | `outputs/two-turn-nc-*/Results/*/result.json` | PNG (hard-coded path) |
| `combined_latency_per_turn_bar.py` | Bar chart of per-turn coder runtime and repetition % across model sizes | `output_*/Results/human_eval/result.json`, console logs | PNG (hard-coded path) |
| `compare_turn_distribution.py` | Stacked bar charts comparing turn distributions across runs | `result.json` files (via `--run` CLI args) | PNG (via `-o` flag) |
| `token_usage_stat.py` | Token usage statistics (prompt, completion, reasoning) and latency across models | `output_*/Results/*/result.json`, console logs | Console tables, PNG |
| `token_usage_pass_failed_stat.py` | Token usage per turn for passed vs failed tasks (2x2 grid: P50/P99 x Pass/Fail) | `output_*/Results/*/result.json`, console logs | PNG (via `-o` flag) |
| `analyze_logprobs_csv.py` | Statistical analysis comparing logprobs distributions (correct vs incorrect) with Welch's t-test | `reports/logprobs.csv` | Console output, optional PNG (via `-o`) |
| `visualize_logprobs.py` | Multi-panel histogram comparison of logprobs distributions with KDE overlays | Logprobs CSV (positional arg) | PNG (via `-o` flag, default: `reports/logprobs_viz.png`) |

### Benchmark runner

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `run_bench.py` | Launches SGLang model servers and runs AgBench benchmarks concurrently | YAML config or CLI args | Benchmark results in configured output dirs |

### Experiment generation

| Script | Description | Input | Output |
|--------|-------------|-------|--------|
| `generate_twoturn_full_matrix.py` | Generates configs + sbatch submission script for the full two-turn-nc matrix (7 turn1 x 10 turn2 models, filtered by same-mode size ordering, minus existing runs) | Hardcoded model list + existing run list | `scripts/configs/config-twoturn-nc-*.yaml`, `scripts/submit_full_matrix.sh` |

## Common data pipeline

```
export_two_turn_csv.py          (outputs/two-turn-* -> reports/csv/two_turn_results.csv)

two_turn_nc_stats.py            (outputs/two-turn-nc-* result.json + console logs -> reports/csv/two_turn_nc_stats.csv)
        |
        v
pareto_latency_accuracy.py      (reports/csv/two_turn_nc_stats.csv -> reports/figures/pareto_latency_accuracy.png)
```
