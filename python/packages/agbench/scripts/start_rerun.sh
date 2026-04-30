#!/bin/bash
# Resume a previously-launched run: fill in only the rep dirs that don't
# exist yet under the existing timestamped Results subdir. Unlike start.sh
# this does NOT wipe $work_dir, and it forces agbench to reuse the existing
# timestamped Results subdir via AGBENCH_RESULTS_DIR_OVERRIDE so its
# per-rep isdir-skip kicks in.
#
# Usage: sbatch scripts/start_rerun.sh <config.yaml> <work_dir>
# Pre-req: any partially-completed (crashed) rep dirs must be removed first;
# agbench skips by directory existence, not by completion.

#SBATCH --job-name=agbench-rerun
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=99:00:00
#SBATCH --partition=gupta
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1

set -euo pipefail

CONFIG="$1"
WORK_DIR="$2"

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate sglang-env

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../.env" ]; then
    set -a; source "$SCRIPT_DIR/../.env"; set +a
fi
export HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN not set. Create python/packages/agbench/.env with HF_TOKEN=...}"

# Auto-discover the existing timestamped Results subdir. Expect exactly one.
RESULTS_PARENT="$WORK_DIR/Results"
if [ ! -d "$RESULTS_PARENT" ]; then
    echo "ERROR: $RESULTS_PARENT does not exist - nothing to resume." >&2
    exit 1
fi
mapfile -t TIMESTAMPED_DIRS < <(find "$RESULTS_PARENT" -mindepth 1 -maxdepth 1 -type d | sort)
if [ "${#TIMESTAMPED_DIRS[@]}" -eq 0 ]; then
    echo "ERROR: no timestamped subdir under $RESULTS_PARENT to resume." >&2
    exit 1
fi
if [ "${#TIMESTAMPED_DIRS[@]}" -gt 1 ]; then
    echo "ERROR: multiple timestamped subdirs under $RESULTS_PARENT - cannot pick:" >&2
    printf '  %s\n' "${TIMESTAMPED_DIRS[@]}" >&2
    exit 1
fi
EXISTING_DIR="${TIMESTAMPED_DIRS[0]}"
export AGBENCH_RESULTS_DIR_OVERRIDE="$EXISTING_DIR"
echo "Resuming into: $AGBENCH_RESULTS_DIR_OVERRIDE"

python scripts/runners/run_bench.py --config "$CONFIG" --work-dir "$WORK_DIR"
