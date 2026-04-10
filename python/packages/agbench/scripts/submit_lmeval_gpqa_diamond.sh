#!/bin/bash
# Submit a GPQA-Diamond evaluation with LMEvalAgent on Qwen3-4B.
#
# Usage:
#   bash scripts/submit_lmeval_gpqa_diamond.sh             # full 198 questions
#   bash scripts/submit_lmeval_gpqa_diamond.sh full        # full 198 questions
#   bash scripts/submit_lmeval_gpqa_diamond.sh smoke       # 4 questions, ~5-10 min
#
# The full run takes ~90-180 minutes wall clock depending on per-request
# thinking length and GPU contention. The smoke run is for verifying the
# pipeline end-to-end (config injection, tool-call parsing, ReAct loop,
# answer extraction, scoring) before committing to the full eval.

set -e
cd "$(dirname "$0")/.."

MODE="${1:-full}"
case "$MODE" in
    full)
        CONFIG="scripts/configs/config-lmeval-gpqa-diamond-4b.yaml"
        WORK_DIR="outputs/lmeval-gpqa-diamond-4b"
        SCENARIO="gpqa_diamond_full"
        TASK="gpqa_diamond_qwen3_4b_full"
        PORT=40006
        ;;
    smoke)
        CONFIG="scripts/configs/config-lmeval-gpqa-diamond-smoke-4b.yaml"
        WORK_DIR="outputs/lmeval-gpqa-diamond-smoke-4b"
        SCENARIO="gpqa_diamond_smoke4"
        TASK="gpqa_diamond_qwen3_4b_smoke4"
        PORT=40007
        ;;
    *)
        echo "Unknown mode: $MODE (use 'full' or 'smoke')" >&2
        exit 2
        ;;
esac

echo "Submitting LMEval GPQA-Diamond ($MODE) on Qwen3-4B (ReAct + tools)..."
echo "  config:   $CONFIG"
echo "  work dir: $WORK_DIR"
echo

sbatch scripts/start_lmeval_long.sh "$CONFIG" "$WORK_DIR"

echo
echo "Submitted."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo
echo "Tail SGLang server log:"
echo "  tail -f $WORK_DIR/logs/sglang_server_$PORT.log"
echo
echo "Tail agbench-run wrapper log:"
echo "  tail -f $WORK_DIR/benchmark_lmeval_gpqa_diamond_*qwen3_4b.log"
echo
echo "Tail per-task scenario.py log (real-time, line-buffered):"
echo "  tail -f $WORK_DIR/Results/*/$SCENARIO/$TASK/0/console_log.txt"
echo
echo "Per-question progress (one line per completed question):"
echo "  grep -c '\\[lmeval_agent_step\\]' $WORK_DIR/Results/*/$SCENARIO/$TASK/0/console_log.txt"
echo
echo "Final results (after job finishes):"
echo "  python -c 'import json; d=json.load(open(\"'\"$WORK_DIR\"'/Results/*/$SCENARIO/$TASK/0/lm_eval_result.json\")); import sys; print(json.dumps(d[\"results\"], indent=2))'"
