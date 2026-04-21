#!/usr/bin/env python3
"""Recompute all logprobs signals from the raw per-token CSV.

Usage:
    python scripts/recompute_signals_from_raw.py reports/csv/logprobs_raw_1.7b_nothink.csv \
        -o reports/csv/logprobs_1.7b_nothink_v2.csv
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd

# Import the shared computation function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_logprobs_csv import FIELDS, compute_token_stats


def main():
    parser = argparse.ArgumentParser(description="Recompute signals from raw per-token CSV.")
    parser.add_argument("raw_csv", help="Path to raw per-token CSV")
    parser.add_argument("-o", "--output", required=True, help="Output summary CSV path")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k for entropy (default: 20)")
    parser.add_argument("--first-n", type=int, default=20, help="First N tokens for short-gen signals")
    args = parser.parse_args()

    print(f"Loading {args.raw_csv} ...")
    df = pd.read_csv(args.raw_csv)
    print(f"  {len(df)} token rows, {df['task_id'].nunique()} tasks")

    rows = []
    grouped = df.groupby(["run", "task_id", "rep", "correct"])
    total = len(grouped)

    for i, ((run, task_id, rep, correct), group) in enumerate(grouped):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{total}...")

        group = group.sort_values("token_idx")

        # Reconstruct the logprobs list format expected by compute_token_stats
        logprobs = []
        for _, tok_row in group.iterrows():
            lp_entry = {
                "token": tok_row["token"],
                "logprob": tok_row["logprob"],
            }
            if pd.notna(tok_row["top_logprobs_json"]) and tok_row["top_logprobs_json"]:
                lp_entry["top_logprobs"] = json.loads(tok_row["top_logprobs_json"])
            logprobs.append(lp_entry)

        stats = compute_token_stats(logprobs, top_k=args.top_k, first_n=args.first_n)
        if not stats:
            continue

        row = {"run": run, "task_id": task_id, "rep": rep, "correct": int(correct)}
        row.update(stats)
        rows.append(row)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} summary rows to {args.output}")


if __name__ == "__main__":
    main()
