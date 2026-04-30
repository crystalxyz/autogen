#!/bin/bash
# Resubmit only HotpotQA SelectorGroupChat runs that have missing trials
# (8b-t: 21 missing, 14b-t: 134 missing as of 2026-04-29). The rerun job
# uses start_rerun.sh which preserves $work_dir and points agbench at the
# existing timestamped Results subdir so already-complete reps are skipped.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CONFIGS=(
  "scripts/configs/config-hotpotqa-selector-8b-t.yaml"
  "scripts/configs/config-hotpotqa-selector-14b-t.yaml"
)

WORK_DIRS=(
  "outputs/hotpotqa-selector-8b-t"
  "outputs/hotpotqa-selector-14b-t"
)

mkdir -p logs

for i in "${!CONFIGS[@]}"; do
  echo "Resubmitting: ${CONFIGS[$i]} -> ${WORK_DIRS[$i]}"
  sbatch scripts/start_rerun.sh "${CONFIGS[$i]}" "${WORK_DIRS[$i]}"
done

echo "All ${#CONFIGS[@]} rerun jobs submitted."
