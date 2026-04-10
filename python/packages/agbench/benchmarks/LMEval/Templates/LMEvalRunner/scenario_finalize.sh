# Sourced by run.sh after scenario.py exits. Deletes everything in the
# task directory except the two artifacts worth keeping:
#   - console_log.txt      (being written to by the parent process)
#   - lm_eval_result.json  (full lm-eval dump, incl. samples + latencies)
#
# Note: scenario_finalize.sh itself is also deleted — safe because bash
# has already sourced it before the find runs.
find . -maxdepth 1 -type f \
    ! -name 'console_log.txt' \
    ! -name 'lm_eval_result.json' \
    -delete 2>/dev/null
