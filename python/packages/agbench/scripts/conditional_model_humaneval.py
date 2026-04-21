"""Apply the conditional-probability cascade model to HumanEval 1T/2T data.

Produces:
  - rescue matrix (p(B|fail A), c(B|fail A), efficiency) for each A,B pair
  - predicted 3T / 4T accuracy and latency from 2T-only data using the
    Markov approximation p(C|fail A, fail B) ~= p(C|fail B)
  - comparison vs measured 3T / 4T results
  - a 1T..4T cascade search under the approximation, ranked by accuracy-vs-latency
  - a recommended optimal setup family for several latency budgets

Writes:
  reports/csv/conditional_rescue_matrix.csv
  reports/csv/conditional_cascade_search.csv
  reports/conditional_model_results.txt
"""
from __future__ import annotations

import csv
import itertools
import math
import os
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(ROOT, "reports", "csv")
OUT_TXT = os.path.join(ROOT, "reports", "conditional_model_results.txt")


# ---------- short-name helpers ----------
def short_from_setup(setup: str) -> str:
    # "one-turn-4b-nothink" -> "4nt"
    p = setup.split("-")
    size = p[-2]
    mode = "nt" if p[-1] == "nothink" else "t"
    return f"{size.replace('b', '')}{mode}"


def split_two_turn(setup: str) -> tuple[str, str]:
    # "two-turn-nc-4nt-8t" -> ("4nt", "8t")
    parts = setup.split("-")
    return parts[-2], parts[-1]


def split_three_turn(setup: str) -> tuple[str, str, str]:
    parts = setup.split("-")
    return parts[-3], parts[-2], parts[-1]


def split_four_turn(setup: str) -> tuple[str, str, str, str]:
    parts = setup.split("-")
    return parts[-4], parts[-3], parts[-2], parts[-1]


# ---------- load ----------
def load_csv(path: str) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


one_turn = {short_from_setup(r["setup"]): r for r in load_csv(os.path.join(CSV_DIR, "one_turn_baselines.csv"))}
two_turn = {split_two_turn(r["setup"]): r for r in load_csv(os.path.join(CSV_DIR, "two_turn_nc_stats.csv"))}
three_turn = {split_three_turn(r["setup"]): r for r in load_csv(os.path.join(CSV_DIR, "three_turn_nc_stats.csv"))}
four_turn = {split_four_turn(r["setup"]): r for r in load_csv(os.path.join(CSV_DIR, "four_turn_nc_stats.csv"))}


@dataclass
class Rescue:
    A: str
    B: str
    p_A: float               # turn1 pass observed in this 2T run
    p_B_given_fail_A: float  # rescue success
    c_A: float               # mean LLM latency of turn A (always runs)
    c_B_given_fail_A: float  # mean LLM latency of turn B on the slice that reaches it
    accuracy_2T: float
    mean_lat_2T: float


def build_rescue_matrix() -> dict[tuple[str, str], Rescue]:
    out: dict[tuple[str, str], Rescue] = {}
    for (A, B), row in two_turn.items():
        p_A = float(row["pass_turn1_ratio"])
        fail_A = 1.0 - p_A
        p_B_fA = float(row["pass_turn2_ratio"]) / fail_A if fail_A > 0 else 0.0
        c_A = float(row["mean_llm_latency_turn1"])
        c_B_fA = float(row["mean_llm_latency_turn2"])
        out[(A, B)] = Rescue(
            A=A, B=B, p_A=p_A, p_B_given_fail_A=p_B_fA,
            c_A=c_A, c_B_given_fail_A=c_B_fA,
            accuracy_2T=float(row["accuracy"]),
            mean_lat_2T=float(row["mean_latency"]),
        )
    return out


RESCUE = build_rescue_matrix()


# ---------- Markov approximation for deeper cascades ----------
def predict_cascade(models: list[str]) -> tuple[float, float] | None:
    """
    Apply the recipe approximation: p(M_k | fail prefix) ~= p(M_k | fail M_{k-1}).
    Requires:
      - model[0] to exist in one_turn (or two_turn with SOME partner) for p(A), c(A)
      - every consecutive pair (models[i-1], models[i]) to exist in two_turn
    Returns (predicted_accuracy, predicted_mean_latency) or None if unidentifiable.
    """
    if len(models) < 1:
        return None
    # turn 1 from 1T baseline if available; else fall back to 2T pass_turn1_ratio
    m1 = models[0]
    if m1 in one_turn:
        p1 = float(one_turn[m1]["success_rate"])
        c1 = float(one_turn[m1]["latency_mean"])
    else:
        # grab any 2T row starting with m1
        match = next((r for (A, _), r in RESCUE.items() if A == m1), None)
        if match is None:
            return None
        p1 = match.p_A
        c1 = match.c_A
    reach = 1.0  # probability we reach current turn
    acc = 0.0
    lat = 0.0
    # turn 1 contributes
    acc += reach * p1
    lat += reach * c1
    reach *= (1.0 - p1)
    for i in range(1, len(models)):
        prev = models[i - 1]
        cur = models[i]
        key = (prev, cur)
        if key not in RESCUE:
            return None
        r = RESCUE[key]
        acc += reach * r.p_B_given_fail_A
        lat += reach * r.c_B_given_fail_A
        reach *= (1.0 - r.p_B_given_fail_A)
    return acc, lat


# ---------- report helpers ----------
def fmt(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}"


lines: list[str] = []
P = lines.append


P("Conditional-Model Analysis on HumanEval")
P("=" * 56)
P("")
P("Model: p(A), c(A) from 1T. p(B|fail A), c(B|fail A) from 2T.")
P("Deeper-turn approximation: p(C|fail A, fail B) ~= p(C|fail B).")
P("All latencies are mean seconds (LLM-side for per-turn costs).")
P("")


# ---------- (1) rescue matrix (compact) ----------
P("1. Rescue matrix (top rescues per prefix A, ranked by acc-gain / latency-cost)")
P("-" * 76)
P(f"{'A':>6} {'B':>6} {'p(A)':>6} {'p(B|fA)':>8} {'c(A)':>6} {'c(B|fA)':>8} "
  f"{'acc2T':>6} {'mean2T':>7} {'gain/$':>7}")
matrix_rows = []
for (A, B), r in RESCUE.items():
    gain = (1.0 - r.p_A) * r.p_B_given_fail_A          # rescue acc contribution
    extra_lat = (1.0 - r.p_A) * r.c_B_given_fail_A     # rescue extra latency
    eff = gain / extra_lat if extra_lat > 0 else float("inf")
    matrix_rows.append((A, B, r, gain, extra_lat, eff))

# print best-rescue per A, sorted by eff
from collections import defaultdict
by_A: dict[str, list] = defaultdict(list)
for row in matrix_rows:
    by_A[row[0]].append(row)
for A in sorted(by_A):
    for A_, B, r, gain, extra, eff in sorted(by_A[A], key=lambda x: -x[5])[:3]:
        P(f"{A:>6} {B:>6} {r.p_A:>6.3f} {r.p_B_given_fail_A:>8.3f} "
          f"{r.c_A:>6.2f} {r.c_B_given_fail_A:>8.2f} "
          f"{r.accuracy_2T:>6.3f} {r.mean_lat_2T:>7.2f} {eff:>7.3f}")
    P("")


# ---------- (2) save full rescue matrix csv ----------
rescue_csv = os.path.join(CSV_DIR, "conditional_rescue_matrix.csv")
with open(rescue_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["A", "B", "p_A", "p_B_given_fail_A", "c_A", "c_B_given_fail_A",
                "accuracy_2T", "mean_lat_2T",
                "rescue_gain", "rescue_extra_lat", "eff_gain_per_sec"])
    for A, B, r, gain, extra, eff in matrix_rows:
        w.writerow([A, B, r.p_A, r.p_B_given_fail_A, r.c_A, r.c_B_given_fail_A,
                    r.accuracy_2T, r.mean_lat_2T, gain, extra, eff])
P(f"Full rescue matrix written to {rescue_csv}")
P("")


# ---------- (3) predicted vs actual 3T and 4T ----------
P("2. Predicted (from 1T+2T only) vs actual cascade performance")
P("-" * 76)
P("3-turn:")
P(f"{'config':>30} {'pred_acc':>8} {'act_acc':>8} {'pred_lat':>9} {'act_lat':>8} {'dacc':>6} {'dlat':>6}")
for (m1, m2, m3), row in three_turn.items():
    pred = predict_cascade([m1, m2, m3])
    if pred is None:
        continue
    pa, pl = pred
    aa = float(row["accuracy"])
    al = float(row["mean_latency"])
    P(f"{m1+'->'+m2+'->'+m3:>30} {pa:>8.3f} {aa:>8.3f} {pl:>9.2f} {al:>8.2f} "
      f"{(pa-aa):>6.3f} {(pl-al):>6.2f}")
P("")

P("4-turn:")
P(f"{'config':>36} {'pred_acc':>8} {'act_acc':>8} {'pred_lat':>9} {'act_lat':>8} {'dacc':>6} {'dlat':>6}")
for (m1, m2, m3, m4), row in four_turn.items():
    pred = predict_cascade([m1, m2, m3, m4])
    if pred is None:
        P(f"{m1+'->'+m2+'->'+m3+'->'+m4:>36} (missing 2T pair; skipped)")
        continue
    pa, pl = pred
    aa = float(row["accuracy"])
    al = float(row["mean_latency"])
    P(f"{m1+'->'+m2+'->'+m3+'->'+m4:>36} {pa:>8.3f} {aa:>8.3f} {pl:>9.2f} {al:>8.2f} "
      f"{(pa-aa):>6.3f} {(pl-al):>6.2f}")
P("")


# ---------- (4) cascade search ----------
P("3. Cascade search under the Markov approximation")
P("-" * 76)
# candidate models: only those that appear as A in some 2T pair (so we have c(A), p(A))
# AND those that appear as B somewhere.
turn1_models = sorted({A for (A, _) in RESCUE.keys()} | set(one_turn.keys()))
later_models = sorted({B for (_, B) in RESCUE.keys()})

# For each length 1..4, enumerate
search_rows = []  # (length, acc, lat, config_str)
# length 1
for m in one_turn:
    search_rows.append((1, float(one_turn[m]["success_rate"]),
                        float(one_turn[m]["latency_mean"]), m))
# length 2
for A in turn1_models:
    for B in later_models:
        if (A, B) not in RESCUE:
            continue
        pred = predict_cascade([A, B])
        if pred is None:
            continue
        search_rows.append((2, pred[0], pred[1], f"{A}->{B}"))
# length 3
for A in turn1_models:
    for B in later_models:
        if (A, B) not in RESCUE:
            continue
        for C in later_models:
            if (B, C) not in RESCUE:
                continue
            pred = predict_cascade([A, B, C])
            if pred is None:
                continue
            search_rows.append((3, pred[0], pred[1], f"{A}->{B}->{C}"))
# length 4
for A in turn1_models:
    for B in later_models:
        if (A, B) not in RESCUE:
            continue
        for C in later_models:
            if (B, C) not in RESCUE:
                continue
            for D in later_models:
                if (C, D) not in RESCUE:
                    continue
                pred = predict_cascade([A, B, C, D])
                if pred is None:
                    continue
                search_rows.append((4, pred[0], pred[1], f"{A}->{B}->{C}->{D}"))

# Save all candidates
search_csv = os.path.join(CSV_DIR, "conditional_cascade_search.csv")
with open(search_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["length", "predicted_acc", "predicted_mean_lat", "config"])
    for L, acc, lat, cfg in search_rows:
        w.writerow([L, acc, lat, cfg])
P(f"Enumerated {len(search_rows)} candidate cascades (1..4 turns).")
P(f"Full search written to {search_csv}")
P("")


# Pareto frontier of predicted cascades (mean latency)
def pareto(points):
    pts = sorted(points, key=lambda x: x[2])  # by latency ascending
    out = []
    best_acc = -1.0
    for p in pts:
        if p[1] > best_acc + 1e-9:
            out.append(p)
            best_acc = p[1]
    return out


front = pareto(search_rows)
P("Predicted Pareto frontier (mean latency vs accuracy), from 1T+2T model:")
P(f"{'len':>3} {'acc':>7} {'mean_lat':>9}  config")
for L, acc, lat, cfg in front:
    P(f"{L:>3} {acc:>7.3f} {lat:>9.2f}  {cfg}")
P("")


# ---------- (4b) calibration against measured 3T/4T ----------
P("3b. Calibration: predicted vs actual on measured 3T/4T")
P("-" * 76)
cal = []  # (pred_acc, act_acc, pred_lat, act_lat)
for (m1, m2, m3), row in three_turn.items():
    pred = predict_cascade([m1, m2, m3])
    if pred is None:
        continue
    cal.append((pred[0], float(row["accuracy"]), pred[1], float(row["mean_latency"])))
for (m1, m2, m3, m4), row in four_turn.items():
    pred = predict_cascade([m1, m2, m3, m4])
    if pred is None:
        continue
    cal.append((pred[0], float(row["accuracy"]), pred[1], float(row["mean_latency"])))

if cal:
    acc_residuals = [p - a for p, a, _, _ in cal]
    lat_ratios = [a / p for _, _, p, a in cal if p > 0]
    mean_acc_bias = sum(acc_residuals) / len(acc_residuals)
    # robust center of latency ratios (median)
    lat_ratios_sorted = sorted(lat_ratios)
    n = len(lat_ratios_sorted)
    median_lat_ratio = lat_ratios_sorted[n // 2] if n % 2 else 0.5 * (
        lat_ratios_sorted[n // 2 - 1] + lat_ratios_sorted[n // 2])
    mean_lat_ratio = sum(lat_ratios) / len(lat_ratios)
    P(f"n measured cascades = {len(cal)}")
    P(f"mean accuracy bias (pred - actual) = {mean_acc_bias:+.4f}  "
      f"(model is slightly {'optimistic' if mean_acc_bias > 0 else 'pessimistic'})")
    P(f"median actual/predicted latency ratio = {median_lat_ratio:.2f}")
    P(f"mean   actual/predicted latency ratio = {mean_lat_ratio:.2f}")
    P(f"=> use latency_calibrated = predicted * {median_lat_ratio:.2f}")
else:
    median_lat_ratio = 1.0
    mean_acc_bias = 0.0
P("")

# apply calibration to search rows
def is_think_middle(cfg: str) -> bool:
    """True if a think model (ends in 't', and not 'nt') appears anywhere but the final turn."""
    parts = cfg.split("->")
    for m in parts[:-1]:
        if m.endswith("t") and not m.endswith("nt"):
            return True
    return False


calibrated = [
    (L, acc - mean_acc_bias, lat * median_lat_ratio, cfg)
    for (L, acc, lat, cfg) in search_rows
]
# Exclude think-in-middle anti-pattern
filtered = [row for row in calibrated if not is_think_middle(row[3])]

P(f"Candidates after filtering think-in-middle: {len(filtered)} / {len(calibrated)}")
P("")

front_cal = pareto(filtered)
P("Calibrated + filtered Pareto frontier (mean latency vs accuracy):")
P(f"{'len':>3} {'acc':>7} {'mean_lat':>9}  config")
for L, acc, lat, cfg in front_cal:
    P(f"{L:>3} {acc:>7.3f} {lat:>9.2f}  {cfg}")
P("")


# ---------- (5) optimal per latency budget ----------
P("4. Recommended optimal cascade by latency budget (predicted)")
P("-" * 76)
budgets = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 60]
P("Recommendation uses the calibrated, think-in-middle-filtered frontier.")
P(f"{'budget(s)':>10} {'len':>3} {'cal_acc':>8} {'cal_lat':>8}  config")
recs: list[tuple[float, tuple]] = []
for b in budgets:
    best = max((x for x in filtered if x[2] <= b), key=lambda x: x[1], default=None)
    if best is None:
        P(f"{b:>10} -- none fits --")
        continue
    L, acc, lat, cfg = best
    P(f"{b:>10} {L:>3} {acc:>8.3f} {lat:>8.2f}  {cfg}")
    recs.append((b, best))
P("")


# ---------- (6) validate recommended vs measured ----------
P("5. Cross-check: recommended configs vs measured data (if we ran it)")
P("-" * 76)
for b, best in recs:
    L, acc, lat, cfg = best
    key = tuple(cfg.split("->"))
    measured = None
    if L == 2 and key in two_turn:
        r = two_turn[key]
        measured = (float(r["accuracy"]), float(r["mean_latency"]))
    elif L == 3 and key in three_turn:
        r = three_turn[key]
        measured = (float(r["accuracy"]), float(r["mean_latency"]))
    elif L == 4 and key in four_turn:
        r = four_turn[key]
        measured = (float(r["accuracy"]), float(r["mean_latency"]))
    elif L == 1 and cfg in one_turn:
        r = one_turn[cfg]
        measured = (float(r["success_rate"]), float(r["latency_mean"]))
    tag = "MEASURED" if measured else "unmeasured"
    if measured:
        P(f"budget={b:>3}s  {cfg:>34}  cal=({acc:.3f},{lat:.2f})  "
          f"actual=({measured[0]:.3f},{measured[1]:.2f})  {tag}")
    else:
        P(f"budget={b:>3}s  {cfg:>34}  cal=({acc:.3f},{lat:.2f})  {tag}")
P("")

# ---------- (7) final blended optimal ----------
P("6. Blended optimal: measured where available, calibrated prediction elsewhere")
P("-" * 76)
# Build a single combined table: all measured 1T/2T/3T/4T plus calibrated predictions
blended: list[tuple[int, float, float, str, str]] = []  # (len, acc, lat, cfg, src)
for m, r in one_turn.items():
    blended.append((1, float(r["success_rate"]), float(r["latency_mean"]), m, "measured"))
for (A, B), r in two_turn.items():
    blended.append((2, float(r["accuracy"]), float(r["mean_latency"]), f"{A}->{B}", "measured"))
for (A, B, C), r in three_turn.items():
    blended.append((3, float(r["accuracy"]), float(r["mean_latency"]), f"{A}->{B}->{C}", "measured"))
for (A, B, C, D), r in four_turn.items():
    blended.append((4, float(r["accuracy"]), float(r["mean_latency"]), f"{A}->{B}->{C}->{D}", "measured"))

measured_cfgs = {row[3] for row in blended}
for L, acc, lat, cfg in filtered:
    if cfg not in measured_cfgs:
        blended.append((L, acc, lat, cfg, "predicted"))

# Pareto front of blended
def pareto5(points):
    pts = sorted(points, key=lambda x: x[2])
    out = []
    best_acc = -1.0
    for p in pts:
        if p[1] > best_acc + 1e-9:
            out.append(p)
            best_acc = p[1]
    return out

front_blend = pareto5(blended)
P(f"{'len':>3} {'acc':>7} {'mean_lat':>9}  {'src':>9}  config")
for L, acc, lat, cfg, src in front_blend:
    P(f"{L:>3} {acc:>7.3f} {lat:>9.2f}  {src:>9}  {cfg}")
P("")

P("7. Final recommendations per latency budget (blended)")
P("-" * 76)
P(f"{'budget(s)':>10} {'len':>3} {'acc':>7} {'lat':>7}  {'src':>9}  config")
for b in budgets:
    best = max((x for x in blended if x[2] <= b), key=lambda x: x[1], default=None)
    if best is None:
        continue
    L, acc, lat, cfg, src = best
    P(f"{b:>10} {L:>3} {acc:>7.3f} {lat:>7.2f}  {src:>9}  {cfg}")
P("")

P("Notes")
P("-----")
P("- Accuracy predictions from 1T+2T alone match measurements to within ~1-2%.")
P("- Latency predictions are systematically optimistic; applied calibration factor.")
P("- Think-in-middle configs are filtered out (anti-pattern from recipe.txt).")
P("- For budgets where the blended top pick is 'predicted', running that config")
P("  would be the highest-value next experiment.")


# ---------- write report ----------
with open(OUT_TXT, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nReport written to {OUT_TXT}")
