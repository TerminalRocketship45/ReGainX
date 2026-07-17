"""
run_stat_significance.py  --  ReGainX Statistical Significance Tests
=====================================================================

Loads the per-episode trial CSVs produced by run_shared_evaluation.py and
runs pairwise statistical significance tests on Pearson r and goal rate.

Requirements:
    results/shared_eval/raw/{policy}_trials.csv  (from run_shared_evaluation.py)

Usage:
    python run_stat_significance.py
    python run_stat_significance.py --metric pearson_r
    python run_stat_significance.py --latex          # print LaTeX table rows

Output:
    Console table of t-test and Wilcoxon p-values for all policy pairs.
    Optional LaTeX rows for copy-paste into the appendix.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_DIR = Path("results/shared_eval/raw")

POLICY_NAMES = [
    "policy_brady_deg_recurrent",
    "policy_deg_recurrent",
    "policy_brady_deg",
    "policy_deg",
    "policy_brady_deg_recurrent_noisy",
    "no_exo",
]

PRIMARY = "policy_brady_deg_recurrent"

COMPARISONS = [
    ("policy_brady_deg_recurrent", "no_exo"),
    ("policy_brady_deg_recurrent", "policy_deg_recurrent"),
    ("policy_brady_deg_recurrent", "policy_brady_deg"),
    ("policy_brady_deg_recurrent", "policy_deg"),
    ("policy_brady_deg_recurrent", "policy_brady_deg_recurrent_noisy"),
]

DISPLAY = {
    "policy_brady_deg_recurrent":       "BD RecPPO",
    "policy_deg_recurrent":             "Deg RecPPO",
    "policy_brady_deg":                 "BD MLP",
    "policy_deg":                       "Deg MLP",
    "policy_brady_deg_recurrent_noisy": "Noisy RecPPO",
    "no_exo":                           "No exoskeleton",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_trials(policy_name: str) -> dict:
    """Load per-episode metrics from CSV. Returns {trial_idx: {metric: value}}."""
    path = RAW_DIR / f"{policy_name}_trials.csv"
    if not path.exists():
        return {}
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["trial_idx"])
            data[idx] = {
                "pearson_r":    float(row.get("pearson_r", "nan") or "nan"),
                "goal_achieved": float(row.get("goal_achieved", "nan") or "nan"),
                "reward":       float(row.get("reward", "nan") or "nan"),
                "severity_bin": int(row.get("severity_bin", -1) or -1),
                "radian_bin":   int(row.get("radian_bin", -1) or -1),
            }
    return data


def paired_arrays(data_a: dict, data_b: dict, metric: str):
    """Return matched arrays for episodes present in both policies."""
    common = sorted(set(data_a) & set(data_b))
    a = np.array([data_a[i][metric] for i in common], dtype=float)
    b = np.array([data_b[i][metric] for i in common], dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    return a[mask], b[mask], len(common)


def sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests(metric: str, latex: bool = False):
    print(f"\n{'='*72}")
    print(f"Statistical Significance Tests  |  Metric: {metric}")
    print(f"All comparisons: BD RecPPO (primary) vs ablation")
    print(f"{'='*72}")

    all_data = {}
    for p in POLICY_NAMES:
        d = load_trials(p)
        if d:
            all_data[p] = d
            print(f"  Loaded {len(d):5d} episodes  ->  {p}")
        else:
            print(f"  [MISSING]  {p}  (run run_shared_evaluation.py first)")

    print()
    header = (f"{'Comparison':45s}  {'n':>5}  "
              f"{'Mean A':>7}  {'Mean B':>7}  {'Δ':>7}  "
              f"{'t-test p':>10}  {'Wilcoxon p':>11}  {'sig':>4}")
    print(header)
    print("-" * len(header))

    rows = []
    for pol_a, pol_b in COMPARISONS:
        if pol_a not in all_data or pol_b not in all_data:
            print(f"  [skip] {DISPLAY.get(pol_a, pol_a)} vs {DISPLAY.get(pol_b, pol_b)} — data missing")
            continue

        a, b, n_common = paired_arrays(all_data[pol_a], all_data[pol_b], metric)

        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        delta  = mean_a - mean_b

        if len(a) < 2:
            print(f"  [skip] too few paired episodes ({len(a)})")
            continue

        t_stat, t_p = ttest_rel(a, b)
        try:
            w_stat, w_p = wilcoxon(a - b, alternative="two-sided")
        except ValueError:
            w_p = float("nan")

        label_a = DISPLAY.get(pol_a, pol_a)
        label_b = DISPLAY.get(pol_b, pol_b)
        comparison = f"{label_a} vs {label_b}"
        row = {
            "comparison": comparison,
            "n": n_common,
            "mean_a": mean_a,
            "mean_b": mean_b,
            "delta": delta,
            "t_p": t_p,
            "w_p": w_p,
            "sig": sig_stars(min(t_p, w_p)),
        }
        rows.append(row)

        print(f"  {comparison:45s}  {n_common:5d}  "
              f"{mean_a:7.4f}  {mean_b:7.4f}  {delta:+7.4f}  "
              f"{t_p:10.2e}  {w_p:11.2e}  {sig_stars(min(t_p, w_p)):>4}")

    # Per-quartile breakdown
    print(f"\n{'='*72}")
    print(f"Per-Severity-Quartile Breakdown  |  Metric: {metric}")
    print(f"{'='*72}")
    qname = ["Q1 mild", "Q2", "Q3", "Q4 severe"]
    for pol_a, pol_b in COMPARISONS[:3]:  # top 3 comparisons
        if pol_a not in all_data or pol_b not in all_data:
            continue
        label_a = DISPLAY.get(pol_a, pol_a)
        label_b = DISPLAY.get(pol_b, pol_b)
        print(f"\n  {label_a} vs {label_b}:")
        for q in range(4):
            common = sorted(set(all_data[pol_a]) & set(all_data[pol_b]))
            a_q = np.array([all_data[pol_a][i][metric]
                            for i in common
                            if all_data[pol_a][i]["severity_bin"] == q], dtype=float)
            b_q = np.array([all_data[pol_b][i][metric]
                            for i in common
                            if all_data[pol_b][i]["severity_bin"] == q], dtype=float)
            # Match by index within quartile
            a_q = a_q[~np.isnan(a_q)]
            b_q = b_q[~np.isnan(b_q)]
            n_q = min(len(a_q), len(b_q))
            if n_q < 2:
                print(f"    {qname[q]}: n={n_q} (skip)")
                continue
            _, t_p = ttest_rel(a_q[:n_q], b_q[:n_q])
            try:
                _, w_p = wilcoxon(a_q[:n_q] - b_q[:n_q], alternative="two-sided")
            except ValueError:
                w_p = float("nan")
            print(f"    {qname[q]:12s}: mean_a={np.mean(a_q[:n_q]):.4f}  "
                  f"mean_b={np.mean(b_q[:n_q]):.4f}  "
                  f"t-test p={t_p:.2e}  Wilcoxon p={w_p:.2e}  {sig_stars(min(t_p, w_p))}")

    if latex:
        print(f"\n{'='*72}")
        print("LaTeX rows (paste into Table~\\ref{tab:app_stats}):")
        print(f"{'='*72}")
        for r in rows:
            t_str = f"{r['t_p']:.2e}" if not np.isnan(r["t_p"]) else "—"
            w_str = f"{r['w_p']:.2e}" if not np.isnan(r["w_p"]) else "—"
            print(f"    {r['comparison']} & Pearson $r$ & {t_str} & {w_str} \\\\")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="pearson_r",
                        choices=["pearson_r", "goal_achieved", "reward"])
    parser.add_argument("--latex", action="store_true",
                        help="Also print LaTeX table rows")
    args = parser.parse_args()

    run_tests(args.metric, latex=args.latex)
    if args.metric == "pearson_r":
        run_tests("goal_achieved", latex=args.latex)
