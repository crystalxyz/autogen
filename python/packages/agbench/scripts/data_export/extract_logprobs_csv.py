#!/usr/bin/env python3
"""Extract logprobs statistics and correctness from agbench runs into a CSV.

Usage:
    python scripts/extract_logprobs_csv.py Results/human_eval_AgentChatLogprobs_*/ -o reports/logprobs.csv
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np


def load_run(run_dir: str):
    """Load result.json and all logprobs files from a run directory."""
    result_path = os.path.join(run_dir, "result.json")
    with open(result_path) as f:
        result = json.load(f)

    task_outcomes = {}
    for tid, task in result["tasks"].items():
        task_outcomes[tid] = {}
        for rid, rep in task["repetitions"].items():
            task_outcomes[tid][rid] = rep["success"]

    # Look for logprobs_output in the run dir, or one level up (depending on agbench nesting)
    candidates = [
        os.path.join(run_dir, "logprobs_output"),
        os.path.join(os.path.dirname(run_dir), "logprobs_output"),
    ]
    logprobs_dir = None
    for c in candidates:
        if os.path.isdir(c):
            logprobs_dir = c
            break

    records = []
    if logprobs_dir:
        for fname in sorted(os.listdir(logprobs_dir)):
            if fname.endswith(".json"):
                with open(os.path.join(logprobs_dir, fname)) as f:
                    rec = json.load(f)
                    rec["_file"] = fname
                    records.append(rec)

    return task_outcomes, records


def parse_label(label: str):
    """Parse label like 'HumanEval_9_rep0' into (task_id, rep)."""
    idx = label.rfind("_rep")
    if idx == -1:
        return label, "0"
    return label[:idx], label[idx + 4:]


def _compute_entropy(top_logprobs: list[dict], top_k: int) -> float | None:
    """Compute entropy from top-k logprobs of a single token position."""
    if not top_logprobs:
        return None
    top_lps = [t["logprob"] for t in top_logprobs[:top_k]]
    top_probs = [math.exp(l) for l in top_lps]
    total = sum(top_probs)
    if total <= 0:
        return None
    normalized = [p / total for p in top_probs]
    return -sum(p * math.log(p + 1e-10) for p in normalized if p > 0)


def _compute_margin(top_logprobs: list[dict]) -> float | None:
    """Gap between top-1 and top-2 logprobs."""
    if not top_logprobs or len(top_logprobs) < 2:
        return None
    return top_logprobs[0]["logprob"] - top_logprobs[1]["logprob"]


def _compute_topk_spread(top_logprobs: list[dict], k: int = 10) -> float | None:
    """Variance of the top-k logprobs (how flat the head of the distribution is)."""
    if not top_logprobs or len(top_logprobs) < 2:
        return None
    lps = [t["logprob"] for t in top_logprobs[:k]]
    return float(np.var(lps))


# Tokens that signal hedging / uncertainty in code generation context
HEDGING_TOKENS = {
    "#", "sorry", "Sorry", "SORRY", "note", "Note", "NOTE",
    "TODO", "todo", "FIXME", "fixme", "pass", "Pass",
    "raise", "Raise", "NotImplemented", "...",
    "I", "Unfortunately", "unfortunately", "cannot", "Cannot",
    "However", "however", "alternatively", "Alternatively",
}


def compute_token_stats(logprobs: list[dict], top_k: int = 5, first_n: int = 20) -> dict:
    """Compute statistics from a list of token logprobs.

    Args:
        top_k: Number of top logprobs to use for entropy computation (default: 5).
        first_n: Number of first tokens for short-generation signals (default: 20).
    """
    if not logprobs:
        return {}

    token_logprobs = [lp["logprob"] for lp in logprobs]
    token_probs = [math.exp(lp) for lp in token_logprobs]

    # Per-token entropy from top_logprobs (using top_k)
    entropies = []
    for lp in logprobs:
        if lp.get("top_logprobs"):
            ent = _compute_entropy(lp["top_logprobs"], top_k)
            if ent is not None:
                entropies.append(ent)

    high_conf_count = sum(1 for p in token_probs if p > 0.9)
    low_conf_count = sum(1 for p in token_probs if p < 0.5)

    n = len(token_logprobs)
    sorted_logprobs = sorted(token_logprobs)

    # Tail metrics: mean of worst 10%, 5%, 1% of tokens (lowest logprobs)
    tail_10_n = max(1, n // 10)
    tail_5_n = max(1, n // 20)
    tail_1_n = max(1, n // 100)
    tail_10_mean = float(np.mean(sorted_logprobs[:tail_10_n]))
    tail_5_mean = float(np.mean(sorted_logprobs[:tail_5_n]))
    tail_1_mean = float(np.mean(sorted_logprobs[:tail_1_n]))

    # Entropy spike counts: tokens with entropy above thresholds
    entropy_spikes_05 = sum(1 for e in entropies if e > 0.5)
    entropy_spikes_10 = sum(1 for e in entropies if e > 1.0)
    entropy_spikes_15 = sum(1 for e in entropies if e > 1.5)
    n_entropy = len(entropies) if entropies else 1  # avoid div by zero

    # ── First-token signals ──
    ft = logprobs[0]
    ft_top = ft.get("top_logprobs", [])
    first_token_entropy = _compute_entropy(ft_top, top_k) if ft_top else None
    first_token_top1_prob = math.exp(ft["logprob"])
    first_token_top1_logprob = ft["logprob"]
    first_token_margin = _compute_margin(ft_top)
    first_token_topk_spread = _compute_topk_spread(ft_top, k=10)

    # ── Short-generation signals (first N tokens) ──
    first_n_actual = min(first_n, n)
    first_n_probs = token_probs[:first_n_actual]
    first_n_logprobs = token_logprobs[:first_n_actual]

    # Entropies for first N tokens
    first_n_entropies = []
    for lp in logprobs[:first_n_actual]:
        if lp.get("top_logprobs"):
            ent = _compute_entropy(lp["top_logprobs"], top_k)
            if ent is not None:
                first_n_entropies.append(ent)

    first_n_mean_entropy = float(np.mean(first_n_entropies)) if first_n_entropies else ""
    first_n_max_entropy = float(np.max(first_n_entropies)) if first_n_entropies else ""
    first_n_entropy_var = float(np.var(first_n_entropies)) if len(first_n_entropies) >= 2 else ""
    first_n_min_prob = float(np.min(first_n_probs))
    first_n_max_prob = float(np.max(first_n_probs))
    first_n_mean_prob = float(np.mean(first_n_probs))
    # Perplexity of partial generation = exp(-mean(logprobs))
    first_n_perplexity = float(math.exp(-np.mean(first_n_logprobs)))

    # ── Aggregated / derived signals ──

    # Entropy trajectory: linear slope of entropy over first N tokens (positive = increasing uncertainty)
    entropy_trajectory_slope = ""
    if len(first_n_entropies) >= 3:
        x = np.arange(len(first_n_entropies))
        slope, _ = np.polyfit(x, first_n_entropies, 1)
        entropy_trajectory_slope = float(slope)

    # Full-sequence entropy trajectory slope
    full_entropy_slope = ""
    if len(entropies) >= 3:
        x = np.arange(len(entropies))
        slope, _ = np.polyfit(x, entropies, 1)
        full_entropy_slope = float(slope)

    # Softmax temperature sensitivity: compare routing decision at T=1 vs T=0.5
    # Measure how much the top-1 probability changes — fragile decisions change a lot
    temp_sensitivity = ""
    if ft_top and len(ft_top) >= 2:
        lps = np.array([t["logprob"] for t in ft_top[:10]])
        # At T=1.0 (original)
        probs_t1 = np.exp(lps)
        probs_t1 = probs_t1 / probs_t1.sum()
        # At T=0.5 (sharper)
        lps_scaled = lps / 0.5
        lps_scaled = lps_scaled - np.max(lps_scaled)  # numerical stability
        probs_t05 = np.exp(lps_scaled)
        probs_t05 = probs_t05 / probs_t05.sum()
        # KL divergence D(T=0.5 || T=1.0) as sensitivity measure
        kl = float(np.sum(probs_t05 * np.log((probs_t05 + 1e-10) / (probs_t1 + 1e-10))))
        temp_sensitivity = kl

    # Hedging token probability: max probability assigned to hedging tokens
    # across top_logprobs of the first token
    hedging_max_prob = 0.0
    hedging_sum_prob = 0.0
    if ft_top:
        for t in ft_top:
            tok = t["token"].strip()
            if tok in HEDGING_TOKENS:
                p = math.exp(t["logprob"])
                hedging_max_prob = max(hedging_max_prob, p)
                hedging_sum_prob += p

    # Margins across first N tokens (mean margin = average decision confidence)
    first_n_margins = []
    for lp in logprobs[:first_n_actual]:
        m = _compute_margin(lp.get("top_logprobs", []))
        if m is not None:
            first_n_margins.append(m)
    first_n_mean_margin = float(np.mean(first_n_margins)) if first_n_margins else ""
    first_n_min_margin = float(np.min(first_n_margins)) if first_n_margins else ""

    return {
        "num_tokens": n,
        "mean_logprob": float(np.mean(token_logprobs)),
        "median_logprob": float(np.median(token_logprobs)),
        "std_logprob": float(np.std(token_logprobs)),
        "min_logprob": float(np.min(token_logprobs)),
        "p10_logprob": float(np.percentile(token_logprobs, 10)),
        "p25_logprob": float(np.percentile(token_logprobs, 25)),
        "mean_prob": float(np.mean(token_probs)),
        "median_prob": float(np.median(token_probs)),
        "min_prob": float(np.min(token_probs)),
        "mean_entropy": float(np.mean(entropies)) if entropies else "",
        "median_entropy": float(np.median(entropies)) if entropies else "",
        "max_entropy": float(np.max(entropies)) if entropies else "",
        "high_conf_ratio": high_conf_count / len(token_probs),
        "low_conf_ratio": low_conf_count / len(token_probs),
        "perplexity": float(math.exp(-np.mean(token_logprobs))),
        "sum_logprob": float(np.sum(token_logprobs)),
        # Tail metrics
        "tail_10pct_mean_logprob": tail_10_mean,
        "tail_5pct_mean_logprob": tail_5_mean,
        "tail_1pct_mean_logprob": tail_1_mean,
        # Entropy spike metrics
        "entropy_spikes_05": entropy_spikes_05,
        "entropy_spikes_10": entropy_spikes_10,
        "entropy_spikes_15": entropy_spikes_15,
        "entropy_spike_05_ratio": entropy_spikes_05 / n_entropy,
        "entropy_spike_10_ratio": entropy_spikes_10 / n_entropy,
        "entropy_spike_15_ratio": entropy_spikes_15 / n_entropy,
        # ── First-token signals ──
        "ft_entropy": first_token_entropy if first_token_entropy is not None else "",
        "ft_top1_prob": first_token_top1_prob,
        "ft_top1_logprob": first_token_top1_logprob,
        "ft_margin": first_token_margin if first_token_margin is not None else "",
        "ft_topk_spread": first_token_topk_spread if first_token_topk_spread is not None else "",
        # ── Short-generation signals (first N tokens) ──
        "fn_mean_entropy": first_n_mean_entropy,
        "fn_max_entropy": first_n_max_entropy,
        "fn_entropy_var": first_n_entropy_var,
        "fn_min_prob": first_n_min_prob,
        "fn_max_prob": first_n_max_prob,
        "fn_mean_prob": first_n_mean_prob,
        "fn_perplexity": first_n_perplexity,
        "fn_mean_margin": first_n_mean_margin,
        "fn_min_margin": first_n_min_margin,
        # ── Aggregated / derived signals ──
        "fn_entropy_slope": entropy_trajectory_slope,
        "full_entropy_slope": full_entropy_slope,
        "ft_temp_sensitivity": temp_sensitivity,
        "ft_hedging_max_prob": hedging_max_prob,
        "ft_hedging_sum_prob": hedging_sum_prob,
    }


FIELDS = [
    "run", "task_id", "rep", "correct",
    "num_tokens", "mean_logprob", "median_logprob", "std_logprob",
    "min_logprob", "p10_logprob", "p25_logprob",
    "mean_prob", "median_prob", "min_prob",
    "mean_entropy", "median_entropy", "max_entropy",
    "high_conf_ratio", "low_conf_ratio", "perplexity", "sum_logprob",
    "tail_10pct_mean_logprob", "tail_5pct_mean_logprob", "tail_1pct_mean_logprob",
    "entropy_spikes_05", "entropy_spikes_10", "entropy_spikes_15",
    "entropy_spike_05_ratio", "entropy_spike_10_ratio", "entropy_spike_15_ratio",
    # First-token signals
    "ft_entropy", "ft_top1_prob", "ft_top1_logprob", "ft_margin", "ft_topk_spread",
    # Short-generation signals (first N tokens)
    "fn_mean_entropy", "fn_max_entropy", "fn_entropy_var",
    "fn_min_prob", "fn_max_prob", "fn_mean_prob", "fn_perplexity",
    "fn_mean_margin", "fn_min_margin",
    # Aggregated / derived signals
    "fn_entropy_slope", "full_entropy_slope",
    "ft_temp_sensitivity", "ft_hedging_max_prob", "ft_hedging_sum_prob",
]


RAW_FIELDS = [
    "run", "task_id", "rep", "correct",
    "token_idx", "token", "logprob", "prob",
    "top_logprobs_json",
]


def write_raw_csv(all_raw_rows: list[dict], output_path: str):
    """Write per-token raw CSV with top-20 logprobs as JSON column."""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(all_raw_rows)
    print(f"Wrote {len(all_raw_rows)} token rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract logprobs + correctness to CSV.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories")
    parser.add_argument("-o", "--output", default="reports/csv/logprobs.csv", help="Summary CSV path")
    parser.add_argument("--raw", default=None,
                        help="Raw per-token CSV path (default: <output>_raw.csv)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Number of top logprobs to use for entropy (default: 20)")
    args = parser.parse_args()

    raw_output = args.raw
    if raw_output is None:
        base, ext = os.path.splitext(args.output)
        raw_output = f"{base}_raw{ext}"

    rows = []
    raw_rows = []

    for run_dir in args.run_dirs:
        if not os.path.isdir(run_dir):
            print(f"Skipping {run_dir}: not a directory")
            continue

        run_name = os.path.basename(run_dir.rstrip("/"))
        task_outcomes, records = load_run(run_dir)
        if not records:
            print(f"Skipping {run_dir}: no logprobs files")
            continue

        for rec in records:
            task_id, rep = parse_label(rec["label"])
            success = task_outcomes.get(task_id, {}).get(rep)
            if success is None:
                print(f"  Warning: no outcome for {task_id} rep{rep}")
                continue

            correct = int(success)

            # Summary stats
            stats = compute_token_stats(rec["logprobs"], top_k=args.top_k)
            if not stats:
                continue
            row = {"run": run_name, "task_id": task_id, "rep": rep, "correct": correct}
            row.update(stats)
            rows.append(row)

            # Raw per-token rows
            for idx, lp in enumerate(rec["logprobs"]):
                top_lps = lp.get("top_logprobs")
                raw_rows.append({
                    "run": run_name,
                    "task_id": task_id,
                    "rep": rep,
                    "correct": correct,
                    "token_idx": idx,
                    "token": lp["token"],
                    "logprob": lp["logprob"],
                    "prob": math.exp(lp["logprob"]),
                    "top_logprobs_json": json.dumps(top_lps) if top_lps else "",
                })

    if not rows:
        print("No data found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} summary rows to {args.output}")

    write_raw_csv(raw_rows, raw_output)


if __name__ == "__main__":
    main()
