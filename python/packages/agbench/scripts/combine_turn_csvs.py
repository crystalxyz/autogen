"""Combine 1T/2T/3T/4T latency+accuracy CSVs into one.

Inputs:
  reports/csv/one_turn_baselines.csv
  reports/csv/two_turn_nc_stats.csv
  reports/csv/three_turn_nc_stats.csv
  reports/csv/four_turn_nc_stats.csv

Output:
  reports/csv/all_turns_combined.csv
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "reports" / "csv"
OUT = CSV_DIR / "all_turns_combined.csv"

COLUMNS = [
    "setup",
    "num_turns",
    "count",
    "accuracy",
    "pass_turn1_ratio",
    "pass_turn2_ratio",
    "pass_turn3_ratio",
    "pass_turn4_ratio",
    "fail_ratio",
    "mean_latency",
    "p50_latency",
    "p90_latency",
    "mean_llm_latency_turn1",
    "mean_llm_latency_turn2",
    "mean_llm_latency_turn3",
    "mean_llm_latency_turn4",
    "mean_e2e_pass_turn1",
    "mean_e2e_pass_turn2",
    "mean_e2e_pass_turn3",
    "mean_e2e_pass_turn4",
    "mean_e2e_fail",
]


def load_one_turn() -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / "one_turn_baselines.csv")
    out = pd.DataFrame({
        "setup": df["setup"],
        "num_turns": 1,
        "count": df["completed_reps"],
        "accuracy": df["success_rate"],
        "pass_turn1_ratio": df["success_rate"],
        "fail_ratio": 1 - df["success_rate"],
        "mean_latency": df["latency_mean"],
        "p50_latency": df["latency_p50"],
        "p90_latency": df["latency_p90"],
        "mean_llm_latency_turn1": df["mean_llm_latency_turn1"],
        "mean_e2e_pass_turn1": df["mean_e2e_pass_turn1"],
        "mean_e2e_fail": df["mean_e2e_fail"],
    })
    return out


def load_n_turn(path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.copy()
    df["num_turns"] = n
    return df


def main() -> None:
    frames = [
        load_one_turn(),
        load_n_turn(CSV_DIR / "two_turn_nc_stats.csv", 2),
        load_n_turn(CSV_DIR / "three_turn_nc_stats.csv", 3),
        load_n_turn(CSV_DIR / "four_turn_nc_stats.csv", 4),
    ]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in COLUMNS:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined = combined[COLUMNS]
    combined.to_csv(OUT, index=False)
    print(f"Wrote {len(combined)} rows -> {OUT}")
    print(combined.groupby("num_turns").size())


if __name__ == "__main__":
    main()
