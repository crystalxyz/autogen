#!/usr/bin/env python3
"""Export per-trial pass/fail turn for Qwen3-4B think and nothink runs to CSV.

For each trial, reports which coder turn it passed on, or 'fail' if it didn't pass.
The coder turn is turns // 2 from result.json (which counts both coder + executor steps).
"""

import json
import os

import pandas as pd

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

dirs = {
    "Qwen3-4B-think": os.path.join(BASE_DIR, "outputs", "output_4b_think"),
    "Qwen3-4B-nothink": os.path.join(BASE_DIR, "outputs", "output_4b_nothink"),
}


def collect_pass_turn_data(output_dir: str) -> list[dict]:
    results_dir = os.path.join(output_dir, "Results")
    rows = []

    # Find all result.json files under Results/
    for scenario_run in sorted(os.listdir(results_dir)):
        result_path = os.path.join(results_dir, scenario_run, "result.json")
        if not os.path.isfile(result_path):
            continue
        with open(result_path) as f:
            data = json.load(f)

        for task_id, task_data in data.get("tasks", {}).items():
            for rep_id, rep in task_data.get("repetitions", {}).items():
                success = rep.get("success", False)
                total_turns = rep.get("turns", 0)
                elapsed = rep.get("elapsed_time", 0.0)
                # result.json counts both coder and executor as separate turns
                coder_turns = total_turns // 2
                # max_turns=12 in scenario.py → 6 coder turns max
                max_coder_turns = 6
                if success:
                    pass_turn = coder_turns  # passed on the last coder turn
                    outcome = "pass"
                else:
                    pass_turn = 0
                    if coder_turns >= max_coder_turns:
                        outcome = "fail_after_6_turns"
                    else:
                        outcome = "fail_before_6_turns"

                rows.append({
                    "task": task_id,
                    "trial": int(rep_id),
                    "success": success,
                    "coder_turns": coder_turns,
                    "pass_turn": pass_turn,
                    "outcome": outcome,
                    "elapsed_time": elapsed,
                })

    return rows


frames = []
for label, path in dirs.items():
    print(f"Collecting {label} from {path} ...")
    rows = collect_pass_turn_data(path)
    df = pd.DataFrame(rows)
    df.insert(0, "mode", label)
    frames.append(df)
    n_pass = df["success"].sum()
    n_fail = len(df) - n_pass
    print(f"  {len(df)} trials, {df['task'].nunique()} tasks, {n_pass} pass, {n_fail} fail")
    print(f"  outcome distribution: {df['outcome'].value_counts().to_dict()}")

combined = pd.concat(frames, ignore_index=True)
combined.sort_values(["mode", "task", "trial"], inplace=True)

out_path = os.path.join(BASE_DIR, "reports", "csv", "qwen3_4b_pass_turn.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
combined.to_csv(out_path, index=False)
print(f"\nWrote {len(combined)} rows to {out_path}")
