#!/bin/bash
# Submit all HotpotQA SelectorGroupChat single-model Qwen3 runs (fullwiki, 100-task sanity).
# Each config uses a distinct sglang port (30011-30018) to avoid collisions if
# two SLURM allocations land on the same node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIGS=(
  "scripts/configs/config-hotpotqa-selector-1.7b-nt.yaml"
  "scripts/configs/config-hotpotqa-selector-1.7b-t.yaml"
  "scripts/configs/config-hotpotqa-selector-4b-nt.yaml"
  "scripts/configs/config-hotpotqa-selector-4b-t.yaml"
  "scripts/configs/config-hotpotqa-selector-8b-nt.yaml"
  "scripts/configs/config-hotpotqa-selector-8b-t.yaml"
  "scripts/configs/config-hotpotqa-selector-14b-nt.yaml"
  "scripts/configs/config-hotpotqa-selector-14b-t.yaml"
)

WORK_DIRS=(
  "outputs/hotpotqa-selector-1.7b-nt"
  "outputs/hotpotqa-selector-1.7b-t"
  "outputs/hotpotqa-selector-4b-nt"
  "outputs/hotpotqa-selector-4b-t"
  "outputs/hotpotqa-selector-8b-nt"
  "outputs/hotpotqa-selector-8b-t"
  "outputs/hotpotqa-selector-14b-nt"
  "outputs/hotpotqa-selector-14b-t"
)

mkdir -p logs

for i in "${!CONFIGS[@]}"; do
  echo "Submitting: ${CONFIGS[$i]} -> ${WORK_DIRS[$i]}"
  sbatch scripts/start.sh "${CONFIGS[$i]}" "${WORK_DIRS[$i]}"
done

echo "All ${#CONFIGS[@]} jobs submitted."
