#!/usr/bin/env python3
"""Export per-trial, per-turn token usage for Qwen3-4B think and nothink runs to CSV."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from token_usage_stat import collect_token_data

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

dirs = {
    "Qwen3-4B-think": os.path.join(BASE_DIR, "outputs", "output_4b_think"),
    "Qwen3-4B-nothink": os.path.join(BASE_DIR, "outputs", "output_4b_nothink"),
}

frames = []
for label, path in dirs.items():
    print(f"Collecting {label} from {path} ...")
    df = collect_token_data(path)
    df.insert(0, "mode", label)
    frames.append(df)
    print(f"  {len(df)} rows, {df['task'].nunique()} tasks, turns 1-{df['turn'].max()}")

import pandas as pd

combined = pd.concat(frames, ignore_index=True)
combined.sort_values(["mode", "task", "trial", "turn"], inplace=True)

out_path = os.path.join(BASE_DIR, "reports", "csv", "qwen3_4b_token_usage.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
combined.to_csv(out_path, index=False)
print(f"\nWrote {len(combined)} rows to {out_path}")
