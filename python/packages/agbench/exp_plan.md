# Two-Turn-NC Full Matrix Experiment Plan

## Goal

Run the full matrix of two-turn-nc HumanEval benchmarks with:
- **Turn 1 models (7):** 0.6nt, 1.7nt, 4nt, 8nt, 14nt, 0.6t, 1.7t
- **Turn 2 models (10):** 0.6nt, 1.7nt, 4nt, 8nt, 14nt, 0.6t, 1.7t, 4t, 8t, 14t
- **Total combinations:** 70

## Filtering Rules

1. **Same-mode ordering:** Within the same thinking mode (nt->nt or t->t), turn 2 must be >= turn 1 in size (no point running a weaker model as a second attempt). This eliminates 11 pairs.
2. **Cross-mode:** All nt->t and t->nt pairs are valid (thinking vs non-thinking have no obvious ordering).

After filtering: **59 valid pairs**

## Existing Runs (21, all excluded)

10 good runs + 11 timing-bug runs (already rerun):

**Good:** 0.6nt-0.6t, 0.6nt-1.7nt, 0.6nt-1.7t, 0.6nt-4t, 0.6nt-8t, 1.7nt-4t, 1.7nt-8t, 4nt-4t, 8nt-4t, 8nt-8t

**Already rerun:** 1.7nt-1.7t, 1.7nt-14t, 4nt-1.7t, 4nt-8t, 4nt-14t, 8nt-1.7t, 8nt-14t, 14nt-1.7t, 14nt-4t, 14nt-8t, 14nt-14t

## Runs Needed (38 new)

### Turn 1 = 0.6nt (5)
- 0.6nt -> 0.6nt, 4nt, 8nt, 14nt, 14t

### Turn 1 = 1.7nt (5)
- 1.7nt -> 1.7nt, 4nt, 8nt, 14nt, 0.6t

### Turn 1 = 4nt (4)
- 4nt -> 4nt, 8nt, 14nt, 0.6t

### Turn 1 = 8nt (3)
- 8nt -> 8nt, 14nt, 0.6t

### Turn 1 = 14nt (2)
- 14nt -> 14nt, 0.6t

### Turn 1 = 0.6t (10)
- 0.6t -> 0.6nt, 1.7nt, 4nt, 8nt, 14nt, 0.6t, 1.7t, 4t, 8t, 14t

### Turn 1 = 1.7t (9)
- 1.7t -> 0.6nt, 1.7nt, 4nt, 8nt, 14nt, 1.7t, 4t, 8t, 14t

## Implementation Plan

1. **Script:** `scripts/generate_twoturn_full_matrix.py`
   - Generates YAML config files for each of the 38 runs
   - Each config uses unique port pairs (auto-assigned starting from 31001)
   - Uses `mem_fraction: 0.95` (consistent with recent configs)
   - Uses `repeat: 2` (consistent with recent configs)
   - Skips all 21 existing runs (10 good + 11 already-rerun timing-bug runs)

2. **Script also generates:** `scripts/submit_full_matrix.sh`
   - Calls `sbatch scripts/start.sh <config> <output_dir>` for each run
   - One sbatch job per combination (each job uses 2 GPUs)

## Model Name Mapping

| Abbreviation | HF Model | enable_thinking |
|-------------|----------|-----------------|
| 0.6nt | Qwen/Qwen3-0.6B | false |
| 1.7nt | Qwen/Qwen3-1.7B | false |
| 4nt | Qwen/Qwen3-4B | false |
| 8nt | Qwen/Qwen3-8B | false |
| 14nt | Qwen/Qwen3-14B | false |
| 0.6t | Qwen/Qwen3-0.6B | true |
| 1.7t | Qwen/Qwen3-1.7B | true |
| 4t | Qwen/Qwen3-4B | true |
| 8t | Qwen/Qwen3-8B | true |
| 14t | Qwen/Qwen3-14B | true |
