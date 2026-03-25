# CSV Export Design

Export parsed benchmark results into two CSV files for downstream analysis.

## Data Sources

Each run has:
- `result.json` — task-level success/turns metadata
- `console_log.txt` per task/trial — step-level latency and token usage

Runs to include:
| Label | Path |
|-------|------|
| 0.6B-NT | `outputs/output_0.6b_nothink` |
| 1.7B-NT | `outputs/output_1.7b_nothink` |
| 4B-NT | `outputs/output_4b_nothink` |
| 8B-NT | `outputs/output_8b_nothink` |
| 0.6B-T | `outputs/output_0.6b_think` |
| 1.7B-T | `outputs/output_1.7b_think` |
| 4B-T | `outputs/output_4b_think` |
| 8B-T | `outputs/output_8b_think` |
| 2T:0.6B-NT/4B-T | `outputs/two-turn-0.6nt-4t` |
| 2T:1.7B-NT/4B-T | `outputs/two-turn-1.7nt-4t` |

## CSV 1: Task-Level Summary

**File:** `reports/task_summary.csv`

One row per (setup, task_id, trial).

| Column | Type | Description |
|--------|------|-------------|
| `setup` | str | Run label (e.g., "0.6B-NT", "2T:0.6B→4B") |
| `task_id` | str | Task identifier (e.g., "HumanEval_0") |
| `trial` | int | Repetition index (0, 1, 2, ...) |
| `success` | bool | Whether the task passed |
| `turns` | int/null | Total turn count from result.json (2, 4, 6, ..., 12) |
| `pass_turn` | int/null | Turn number where it passed (1, 2, 3, ...), null if failed |
| `e2e_latency_s` | float/null | Sum of all step latencies from console_log.txt |
| `total_prompt_tokens` | int/null | Sum of prompt tokens across all steps |
| `total_completion_tokens` | int/null | Sum of completion tokens across all steps |
| `total_reasoning_tokens` | int/null | Sum of reasoning tokens across all steps |
| `num_steps` | int/null | Number of runtime entries in the log |

### Derived fields (computed from above)
- `pass_turn` = `turns / 2` if `success` else null (each coder+executor pair = 1 turn)
- `e2e_latency_s` = sum of all `[runtime]` entries in console_log.txt
- Token totals = sum of all `CompletionUsage` entries in console_log.txt

## CSV 2: Step-Level Detail

**File:** `reports/step_detail.csv`

One row per step within a (setup, task_id, trial).

| Column | Type | Description |
|--------|------|-------------|
| `setup` | str | Run label |
| `task_id` | str | Task identifier |
| `trial` | int | Repetition index |
| `step_index` | int | Sequential position in the log (0, 1, 2, ...) |
| `step_name` | str | Agent name from `[runtime]` (e.g., "coder", "executor", "coder_small") |
| `turn_number` | int | Logical turn (1, 2, 3, ...) — `step_index // 2 + 1` |
| `role` | str | "coder" or "executor" — derived from step_name |
| `latency_s` | float | Step latency from `[runtime]` |
| `prompt_tokens` | int/null | From `CompletionUsage` (coder steps only) |
| `completion_tokens` | int/null | From `CompletionUsage` (coder steps only) |
| `reasoning_tokens` | int/null | From `CompletionUsage` (coder steps only) |

### Notes
- Token usage is only available for coder steps (LLM calls), not executor steps.
- `CompletionUsage` entries and `[runtime]` entries are matched by order of appearance.
- For two-turn runs, step_name distinguishes the model: `coder_small` (T1) vs `coder_large` (T2).

## Implementation Plan

1. **Script:** `scripts/export_csv.py`
2. **Approach:**
   - For each run, locate `result.json` and the console_log directory.
   - Parse `result.json` for task-level metadata (success, turns).
   - Parse each `console_log.txt` for step latencies (reuse `parse_step_latencies` from `token_usage_stat.py`) and token usage (reuse `parse_token_usage_from_console_log`).
   - Join task-level and step-level data.
   - Write both CSVs.
3. **Dependencies:** Reuses parsing functions from `token_usage_stat.py`.
