#!/bin/bash
# Quick test script for multi-turn verification

set -e

echo "=========================================="
echo "Multi-Turn HumanEval Test Script"
echo "=========================================="
echo ""

# Change to agbench directory
cd /home/x-yli19/crystal/autogen/python/packages/agbench

# Default to 2-problem test
TEST_FILE="${1:-benchmarks/HumanEval/Tasks/human_eval_CoderVerifier_2_test.jsonl}"

echo "Test file: $TEST_FILE"
echo ""

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    echo ""
    echo "Available test files:"
    ls -lh benchmarks/HumanEval/Tasks/*test*.jsonl 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Count problems
NUM_PROBLEMS=$(wc -l < "$TEST_FILE")
echo "Number of problems: $NUM_PROBLEMS"
echo ""

# Estimate time
ESTIMATED_MINUTES=$((NUM_PROBLEMS * 3))
echo "Estimated time: ~$ESTIMATED_MINUTES minutes (assuming 3 min/problem with multi-turn)"
echo ""

# Ask for confirmation
read -p "Run test? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Test cancelled."
    exit 0
fi

echo ""
echo "Starting test run..."
echo "=========================================="
echo ""

# Run the test
python -m agbench run "$TEST_FILE" --native

# Get the most recent results directory
RESULTS_DIR=$(ls -td Results/human_eval_CoderVerifier_2_test_* 2>/dev/null | head -1)

if [ -z "$RESULTS_DIR" ]; then
    # Try without timestamp suffix
    RESULTS_DIR=$(ls -td Results/*$(basename "$TEST_FILE" .jsonl)* 2>/dev/null | head -1)
fi

if [ -z "$RESULTS_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "Test completed, but couldn't find results directory"
    echo "Check Results/ manually"
    exit 0
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""

# Analyze results
echo "Analyzing multi-turn behavior..."
echo "----------------------------------------"
echo ""

for task_dir in "$RESULTS_DIR"/*_HumanEval_*/; do
    if [ -d "$task_dir" ]; then
        task_name=$(basename "$task_dir")

        # Count agent invocations
        if [ -f "$task_dir/latency.json" ]; then
            total_turns=$(grep -c '"agent_name"' "$task_dir/latency.json" || echo "0")
            coder_turns=$(grep '"agent_name": "Coder"' "$task_dir/latency.json" | wc -l || echo "0")
            verifier_turns=$(grep '"agent_name": "Verifier"' "$task_dir/latency.json" | wc -l || echo "0")
            executor_turns=$(grep '"agent_name": "Executor"' "$task_dir/latency.json" | wc -l || echo "0")

            echo "Task: $task_name"
            echo "  Total turns: $total_turns"
            echo "  Coder: $coder_turns | Verifier: $verifier_turns | Executor: $executor_turns"

            # Check for protocol keywords in console log
            if [ -f "$task_dir/console_log.txt" ]; then
                has_clarification=$(grep -c "REQUEST_CLARIFICATION" "$task_dir/console_log.txt" || echo "0")
                has_changes=$(grep -c "REQUEST_CHANGES" "$task_dir/console_log.txt" || echo "0")
                has_adversarial=$(grep -c "ADVERSARIAL_TEST" "$task_dir/console_log.txt" || echo "0")
                has_approved=$(grep -c "APPROVED" "$task_dir/console_log.txt" || echo "0")

                echo "  Protocol phases:"
                echo "    REQUEST_CLARIFICATION: $has_clarification"
                echo "    REQUEST_CHANGES: $has_changes"
                echo "    ADVERSARIAL_TEST: $has_adversarial"
                echo "    APPROVED: $has_approved"
            fi
            echo ""
        else
            echo "Task: $task_name (no latency data)"
            echo ""
        fi
    fi
done

echo "----------------------------------------"
echo ""
echo "To view detailed conversation for a task:"
echo "  cat $RESULTS_DIR/[task_name]/console_log.txt"
echo ""
echo "To view latency metrics:"
echo "  cat $RESULTS_DIR/[task_name]/latency.json | python -m json.tool"
echo ""
