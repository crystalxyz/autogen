# Sourced by run.sh after scenario.py exits. Deletes everything in the
# task directory except the two artifacts worth keeping:
#   - console_log.txt      (being written to by the parent process)
#   - lm_eval_result.json  (full lm-eval dump, incl. samples)
find . -maxdepth 1 -type f \
    ! -name 'console_log.txt' \
    ! -name 'lm_eval_result.json' \
    -delete 2>/dev/null
