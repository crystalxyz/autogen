#!/bin/bash
# Launch HumanEval AgentChat sweep across Gemma 3 1B / 4B / 12B.
#
# Each sbatch job:
#   1. Launches its own SGLang server on the port declared in its config
#   2. Runs agbench on the HumanEval AgentChat scenario with repeat=5
#   3. Writes results to outputs/humaneval-gemma3-<size>/
#
# Usage:  bash scripts/submit_humaneval_gemma3.sh
#
# To rerun a single model:
#   sbatch scripts/start.sh scripts/configs/config-gemma-4b.yaml outputs/humaneval-gemma3-4b

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIGS=(
  "1b"
  "4b"
  "12b"
)

for size in "${CONFIGS[@]}"; do
  cfg="scripts/configs/config-gemma-${size}.yaml"
  outdir="outputs/humaneval-gemma3-${size}"
  if [ ! -f "$cfg" ]; then
    echo "Skipping: $cfg not found"
    continue
  fi
  echo "Submitting gemma3-${size} -> $outdir"
  sbatch scripts/start.sh "$cfg" "$outdir"
done

echo
echo "Submitted. Monitor with: squeue -u \$USER"
