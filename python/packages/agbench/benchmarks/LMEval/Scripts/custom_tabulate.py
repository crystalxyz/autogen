"""Custom tabulator for LMEval.

Overrides the default timer regex so that "LMEval execution time: ..."
lines in console_log.txt are recognised (the default regex is hardcoded
to "AgentChat execution time: ..." in agbench.tabulate_cmd).
"""

import sys

from agbench.tabulate_cmd import default_tabulate, default_timer


LMEVAL_TIMER_REGEX = r"LMEval execution time:\s*([\d.]+)(?:\s*seconds)?(?:\s*!#!#)?"


def lmeval_timer(instance_dir: str):
    return default_timer(instance_dir, timer_regex=LMEVAL_TIMER_REGEX)


def main(args):
    default_tabulate(args, timer=lmeval_timer)


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
