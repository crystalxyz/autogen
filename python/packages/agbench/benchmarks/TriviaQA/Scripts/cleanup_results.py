#!/usr/bin/env python3
"""Remove all files except console_log.txt and prompt.txt from TriviaQA result directories."""

import argparse
import os

KEEP = {"console_log.txt", "prompt.txt"}


def cleanup(results_dir: str, dry_run: bool = False) -> None:
    removed = 0
    for root, dirs, files in os.walk(results_dir):
        for f in files:
            if f not in KEEP:
                path = os.path.join(root, f)
                if dry_run:
                    print(f"would remove: {path}")
                else:
                    os.remove(path)
                removed += 1
    action = "would remove" if dry_run else "removed"
    print(f"{action} {removed} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up TriviaQA result directories.")
    parser.add_argument("results_dir", help="Path to the results directory")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be removed without deleting")
    args = parser.parse_args()
    cleanup(args.results_dir, dry_run=args.dry_run)
