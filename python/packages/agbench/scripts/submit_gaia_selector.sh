#!/bin/bash
# Submit all 10 GAIA SelectorGroupChat single-model Qwen3 runs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIGS=(

  # "scripts/configs/config-gaia-selector-1.7b-nt.yaml"
  # "scripts/configs/config-gaia-selector-1.7b-t.yaml"
  # "scripts/configs/config-gaia-selector-4b-nt.yaml"
  # "scripts/configs/config-gaia-selector-4b-t.yaml"
  # "scripts/configs/config-gaia-selector-8b-nt.yaml"
  # "scripts/configs/config-gaia-selector-8b-t.yaml"
  # "scripts/configs/config-gaia-selector-14b-nt.yaml"
  "scripts/configs/config-gaia-selector-14b-t.yaml"
)

WORK_DIRS=(
  # "outputs/gaia-selector-1.7b-nt"
  # "outputs/gaia-selector-1.7b-t"
  # "outputs/gaia-selector-4b-nt"
  # "outputs/gaia-selector-4b-t"
  # "outputs/gaia-selector-8b-nt"
  # "outputs/gaia-selector-8b-t"
  # "outputs/gaia-selector-14b-nt"
  "outputs/gaia-selector-14b-t"
)

mkdir -p logs

for i in "${!CONFIGS[@]}"; do
  echo "Submitting: ${CONFIGS[$i]} -> ${WORK_DIRS[$i]}"
  sbatch scripts/start.sh "${CONFIGS[$i]}" "${WORK_DIRS[$i]}"
done

echo "All 10 jobs submitted."
