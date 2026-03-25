import os
import re
import sys

from agbench.tabulate_cmd import default_tabulate


def scorer(instance_dir):
    """
    Score a SWE-bench task instance.

    Success is determined by whether the FAIL_TO_PASS tests passed after
    the agent's fix. The scenario.py prints "ALL TESTS PASSED !#!#" on success.
    """
    console_log_file = os.path.join(instance_dir, "console_log.txt")
    if not os.path.isfile(console_log_file):
        return None

    with open(console_log_file, "rt") as fh:
        console_log = fh.read()

    # Check if scenario completed
    if "SCENARIO.PY COMPLETE !#!#" not in console_log:
        return None

    # Check for success marker
    return "ALL TESTS PASSED !#!#" in console_log


def main(args):
    default_tabulate(args, scorer=scorer)


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
