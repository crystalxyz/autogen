#!/bin/bash
#
# SBATCH wrapper for long-running LMEval benchmarks (full-set evaluations
# with many questions per scenario.py invocation). Differs from start.sh in
# that it raises agbench's per-task timeout from the default 20 minutes to
# 4 hours via AGBENCH_TASK_TIMEOUT, and forces unbuffered Python stdout so
# per-request progress shows up in console_log.txt in real time instead of
# being lost when the process is killed.
#
# Usage:
#   sbatch scripts/start_lmeval_long.sh \
#       scripts/configs/config-lmeval-gpqa-diamond-4b.yaml \
#       outputs/lmeval-gpqa-diamond-4b
#
#SBATCH --job-name=lmeval
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --time=99:00:00
#SBATCH --partition=gupta
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1

# --- Hard cap on a single agbench scenario.py invocation -----------------
# Default in src/agbench/execution.py is 1200s (20 min), which is fine for
# the agentic benchmarks where one scenario = one HumanEval problem, but
# nowhere near enough for an LMEval scenario that internally walks 198
# gpqa_diamond questions. 4 hours covers Qwen3-4B + GPQA-diamond + tool
# overhead with comfortable margin.
export AGBENCH_TASK_TIMEOUT=14400

# --- Force unbuffered Python output --------------------------------------
# scenario.py's per-request prints (`========== REQUEST n ==========`,
# `[lmeval_agent_step] ...`) go to stdout. When stdout is a pipe (which it
# is here, since run.sh captures it into console_log.txt), CPython
# block-buffers in 4-8 KB chunks. If the process is killed mid-run those
# buffers are lost and the log shows zero progress. -u disables that.
export PYTHONUNBUFFERED=1

# --- Conda env -----------------------------------------------------------
source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate sglang-env

# --- Output dir reset (matches start.sh's convention) --------------------
rm -rf "$2"

# --- HF auth (mirrors start.sh) ------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/../../.env" ]; then
    set -a; source "$SCRIPT_DIR/../../.env"; set +a
fi
export HF_TOKEN="${HF_TOKEN:?Error: HF_TOKEN not set. Create python/packages/agbench/.env with HF_TOKEN=...}"

# --- Run -----------------------------------------------------------------
echo "AGBENCH_TASK_TIMEOUT=$AGBENCH_TASK_TIMEOUT"
echo "PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
echo "HF_TOKEN=${HF_TOKEN:0:7}...${HF_TOKEN: -4}"
echo "Config: $1"
echo "Work dir: $2"
echo

python -u scripts/runners/run_bench.py --config "$1" --work-dir "$2"
