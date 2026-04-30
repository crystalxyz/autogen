#!/bin/bash
# Launch the full HotpotQA SelectorGroupChat sweep across 8 Qwen3 variants
# (1.7B/4B/8B/14B × non-thinking/thinking), each on the seeded sanity100 subset
# with 3 repetitions per task.
#
# Each sbatch job:
#   1. Launches its own SGLang server on the port declared in its config
#   2. Runs agbench on the 100-task scenario with repeat=3
#   3. Writes results to outputs/hotpotqa-selector-<size>-<mode>/
#
# Usage:  bash scripts/submit_hotpotqa_sweep.sh
#
# To rerun a single model:
#   sbatch scripts/start.sh scripts/configs/config-hotpotqa-selector-4b-t.yaml outputs/hotpotqa-selector-4b-t

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  "1.7b-nt"
  "1.7b-t"
  "4b-nt"
  "4b-t"
  "8b-nt"
  "8b-t"
  "14b-nt"
  "14b-t"
)

for model in "${CONFIGS[@]}"; do
  cfg="scripts/configs/config-hotpotqa-selector-${model}.yaml"
  outdir="outputs/hotpotqa-selector-${model}"
  if [ ! -f "$cfg" ]; then
    echo "Skipping: $cfg not found"
    continue
  fi
  echo "Submitting $model -> $outdir"
  sbatch scripts/start.sh "$cfg" "$outdir"
done

echo
echo "Submitted. Monitor with: squeue -u \$USER"
