"""
run_impedance_baseline.py  --  ReGainX Impedance-Control Baseline Evaluation
=============================================================================

Evaluates the velocity-deficit impedance controller (Gasperina et al., 2021)
on the same shared 4,000-episode evaluation set used for all RL policies.

Requires:
    results/shared_eval/episode_configs.csv   (from run_shared_evaluation.py)
    policies/healthy_policy.zip
    policies/policy_brady_deg_recurrent.zip   (for significance test)

Outputs:
    results/shared_eval/raw/impedance_baseline_trials.csv
    results/shared_eval/summaries/impedance_baseline_per_quartile.csv
    results/shared_eval/summaries/impedance_baseline_per_radian.csv
    results/shared_eval/plots/comparison_pearsonr_with_impedance.png
    results/shared_eval/plots/pearsonr_by_severity_with_impedance.png
    Console: full summary table + significance vs BD RecPPO

Usage:
    python run_impedance_baseline.py
    python run_impedance_baseline.py --episodes 400   # quick smoke test
"""

import argparse
import csv
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, pearsonr

import myosuite  # noqa: F401 — registers envs
from myosuite.utils import gym
from stable_baselines3 import PPO

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.impedance_baseline import ImpedanceBaseline
from utils import compute_severity

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT          = Path(__file__).parent
POLICIES_DIR  = ROOT / "policies"
OUT_DIR       = ROOT / "results" / "shared_eval"
OUT_RAW       = OUT_DIR / "raw"
OUT_SUMMARIES = OUT_DIR / "summaries"
OUT_PLOTS     = OUT_DIR / "plots"

HEALTHY_PATH = POLICIES_DIR / "healthy_policy.zip"
CONFIGS_CSV  = OUT_DIR / "episode_configs.csv"
RECPPO_CSV   = OUT_RAW / "policy_brady_deg_recurrent_trials.csv"

MAX_STEPS     = 200
SEVERITY_BINS = 4
RADIAN_BINS   = 4
SEV_LABELS    = ["Q1 mild", "Q2", "Q3", "Q4 severe"]
RADIAN_LABELS = ["0.0–0.5 rad", "0.5–1.0 rad", "1.0–1.5 rad", "1.5+ rad"]

TRIAL_FIELDS = [
    "trial_idx", "radian_bin", "severity_bin",
    "start_angle", "target_angle", "radian_travelled",
    "force_scale", "activation_slowdown", "avg_mf", "severity",
    "pearson_r", "reward", "goal_achieved", "goal_time_s",
    "episode_steps", "healthy_reward",
]

# Display names for comparison plots
POLICY_DISPLAY = {
    "no_exo":                           ("No Exo",       "gray",      "x--"),
    "policy_deg":                       ("Deg MLP",      "#1f77b4",   "s:"),
    "policy_brady_deg":                 ("BD MLP",       "#aec7e8",   "s-"),
    "policy_deg_recurrent":             ("Deg RecPPO",   "#ff7f0e",   "o:"),
    "policy_brady_deg_recurrent":       ("BD RecPPO",    "#d62728",   "o-"),
    "policy_brady_deg_recurrent_noisy": ("Noisy RecPPO", "#9467bd",   "^-"),
    "impedance_baseline":               ("Impedance",    "#2ca02c",   "D-"),
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(path: Path) -> list:
    configs = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mf_vals    = np.array([float(v) for v in row["mf_vals"].split(";")])
            split_vals = np.array([float(v) for v in row["split_vals"].split(";")])
            hr = row.get("healthy_reward", "")
            configs.append({
                "trial_idx":           int(row["trial_idx"]),
                "severity_bin":        int(row["severity_bin"]),
                "radian_bin":          int(row["radian_bin"]),
                "start_angle":         float(row["start_angle"]),
                "target_angle":        float(row["target_angle"]),
                "radian_travelled":    float(row["radian_travelled"]),
                "force_scale":         float(row["force_scale"]),
                "activation_slowdown": float(row["activation_slowdown"]),
                "avg_mf":              float(row["avg_mf"]),
                "severity":            float(row["severity"]),
                "healthy_reward":      float(hr) if hr else float("nan"),
                "mf_vals":             mf_vals,
                "split_vals":          split_vals,
            })
    return configs


# ---------------------------------------------------------------------------
# Healthy reference
# ---------------------------------------------------------------------------

def run_healthy_reference(configs: list) -> list:
    """Run healthy policy on every config; returns list of angle trajectories."""
    policy    = PPO.load(str(HEALTHY_PATH))
    obs_dim   = policy.observation_space.shape[0]
    env       = gym.make("myoElbowPose1D6MRandom-v0")
    results   = []
    t0        = time.time()

    print(f"  Running healthy reference on {len(configs)} episodes…")
    for i, cfg in enumerate(configs):
        env.reset()
        u = env.unwrapped
        u.target_jnt_value = [cfg["target_angle"]]
        u.target_type      = "fixed"
        u.update_target(restore_sim=True)
        u.sim.data.qpos[0] = cfg["start_angle"]
        u.sim.data.qvel[:] = 0.0
        u.sim.forward()

        obs    = u.get_obs()[:obs_dim]
        angles = []
        for _ in range(MAX_STEPS):
            action, _ = policy.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)
            obs = obs[:obs_dim]
            angles.append(float(u.sim.data.qpos[0]))
            if done or truncated:
                break
        results.append(angles)

        if (i + 1) % 500 == 0 or (i + 1) == len(configs):
            print(f"    [{i+1:4d}/{len(configs)}]  elapsed={time.time()-t0:.0f}s")

    env.close()
    return results


# ---------------------------------------------------------------------------
# Exo environment helpers
# ---------------------------------------------------------------------------

def make_exo_env() -> CombinedExoOnlyWrapper:
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    return CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=str(HEALTHY_PATH),
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )


def configure_exo_replay(exo_env: CombinedExoOnlyWrapper, cfg: dict) -> None:
    buw = exo_env.base_env.unwrapped
    buw.target_jnt_value = [cfg["target_angle"]]
    buw.target_type      = "fixed"
    buw.update_target(restore_sim=True)

    mf, sp = cfg["mf_vals"], cfg["split_vals"]
    rem    = 1.0 - mf
    buw.muscle_fatigue.MA[:] = rem * sp
    buw.muscle_fatigue.MR[:] = rem * (1.0 - sp)
    buw.muscle_fatigue.MF[:] = mf

    exo_env.force_scale         = cfg["force_scale"]
    exo_env.activation_slowdown = cfg["activation_slowdown"]
    exo_env._apply_brady()

    buw.sim.data.qpos[0] = cfg["start_angle"]
    buw.sim.data.qvel[:] = 0.0
    buw.sim.forward()


def _safe_r(a: list, b: list) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    r, _ = pearsonr(a[:n], b[:n])
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Impedance evaluation
# ---------------------------------------------------------------------------

def run_impedance_eval(configs: list, healthy_results: list) -> list:
    exo_env    = make_exo_env()
    controller = ImpedanceBaseline()
    trials     = []
    t0         = time.time()
    csv_path   = OUT_RAW / "impedance_baseline_trials.csv"

    print(f"\n  Evaluating: impedance_baseline  ({len(configs)} episodes)")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        writer.writeheader()

        for i, (cfg, h_angles) in enumerate(zip(configs, healthy_results)):
            exo_env.reset()
            configure_exo_replay(exo_env, cfg)
            controller.reset()

            angles        = []
            total_reward  = 0.0
            goal_achieved = False
            goal_time     = float("nan")

            for step in range(MAX_STEPS):
                obs    = exo_env._build_obs(exo_env._current_raw_obs())
                action = controller.predict(obs)
                obs, rwd, done, truncated, _ = exo_env.step(action)
                total_reward += float(rwd)
                angles.append(float(exo_env.base_env.unwrapped.sim.data.qpos[0]))

                solved = exo_env.base_env.unwrapped.rwd_dict.get("solved", False)
                if bool(np.asarray(solved).flat[0]) and not goal_achieved:
                    goal_achieved = True
                    goal_time     = step * exo_env.base_env.unwrapped.dt
                if done or truncated:
                    break

            pr  = _safe_r(angles, h_angles)
            row = {
                "trial_idx":           cfg["trial_idx"],
                "radian_bin":          cfg["radian_bin"],
                "severity_bin":        cfg["severity_bin"],
                "start_angle":         f"{cfg['start_angle']:.6f}",
                "target_angle":        f"{cfg['target_angle']:.6f}",
                "radian_travelled":    f"{cfg['radian_travelled']:.6f}",
                "force_scale":         f"{cfg['force_scale']:.6f}",
                "activation_slowdown": f"{cfg['activation_slowdown']:.6f}",
                "avg_mf":              f"{cfg['avg_mf']:.6f}",
                "severity":            f"{cfg['severity']:.6f}",
                "pearson_r":           f"{pr:.6f}",
                "reward":              f"{total_reward:.6f}",
                "goal_achieved":       int(goal_achieved),
                "goal_time_s":         f"{goal_time:.6f}" if not np.isnan(goal_time) else "",
                "episode_steps":       len(angles),
                "healthy_reward":      (f"{cfg['healthy_reward']:.6f}"
                                        if not np.isnan(cfg["healthy_reward"]) else ""),
            }
            writer.writerow(row)
            f.flush()

            trials.append({
                "trial_idx":    cfg["trial_idx"],
                "severity_bin": cfg["severity_bin"],
                "radian_bin":   cfg["radian_bin"],
                "pearson_r":    pr,
                "reward":       total_reward,
                "goal_achieved": int(goal_achieved),
            })

            if (i + 1) % 500 == 0 or (i + 1) == len(configs):
                elapsed = time.time() - t0
                print(f"    [{i+1:4d}/{len(configs)}]  r={pr:.3f}  "
                      f"goal={int(goal_achieved)}  elapsed={elapsed:.0f}s")

    exo_env.close()
    print(f"  Saved → {csv_path}")
    return trials


# ---------------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------------

def save_per_quartile(trials: list) -> list:
    rows = []
    for q in range(SEVERITY_BINS):
        sub = [t for t in trials if t["severity_bin"] == q]
        r_v = [t["pearson_r"] for t in sub]
        g_v = [t["goal_achieved"] for t in sub]
        w_v = [t["reward"] for t in sub]
        rows.append({
            "severity_quartile": SEV_LABELS[q],
            "n":                 len(sub),
            "pearson_r_mean":    round(float(np.mean(r_v)),  4) if r_v else float("nan"),
            "pearson_r_std":     round(float(np.std(r_v)),   4) if r_v else float("nan"),
            "goal_rate":         round(float(np.mean(g_v)),  4) if g_v else float("nan"),
            "reward_mean":       round(float(np.mean(w_v)),  2) if w_v else float("nan"),
        })
    path = OUT_SUMMARIES / "impedance_baseline_per_quartile.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def save_per_radian(trials: list) -> list:
    rows = []
    for b in range(RADIAN_BINS):
        sub = [t for t in trials if t["radian_bin"] == b]
        r_v = [t["pearson_r"] for t in sub]
        g_v = [t["goal_achieved"] for t in sub]
        rows.append({
            "radian_bin":     RADIAN_LABELS[b],
            "n":              len(sub),
            "pearson_r_mean": round(float(np.mean(r_v)), 4) if r_v else float("nan"),
            "goal_rate":      round(float(np.mean(g_v)), 4) if g_v else float("nan"),
        })
    path = OUT_SUMMARIES / "impedance_baseline_per_radian.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# Statistical significance
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def load_csv_r(path: Path) -> dict:
    """Load {trial_idx: pearson_r} from a trials CSV."""
    if not path.exists():
        return {}
    data = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                data[int(row["trial_idx"])] = float(row["pearson_r"])
            except (KeyError, ValueError):
                pass
    return data


def print_significance(imp_trials: list, ref_r_by_idx: dict, ref_label: str) -> None:
    imp_by_idx = {t["trial_idx"]: t["pearson_r"] for t in imp_trials}
    common     = sorted(set(imp_by_idx) & set(ref_r_by_idx))
    if len(common) < 2:
        print("  Not enough matched episodes for significance test.")
        return

    a    = np.array([ref_r_by_idx[i] for i in common])
    b    = np.array([imp_by_idx[i]   for i in common])
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]

    _, t_p = ttest_rel(a, b)
    try:
        _, w_p = wilcoxon(a - b, alternative="two-sided")
    except ValueError:
        w_p = float("nan")

    print(f"\n  {ref_label} vs Impedance Baseline  (n={len(a)} paired episodes):")
    print(f"    {ref_label:<26}: mean r = {np.mean(a):.4f} ± {np.std(a):.4f}")
    print(f"    {'Impedance Baseline':<26}: mean r = {np.mean(b):.4f} ± {np.std(b):.4f}")
    print(f"    Δ (ref − impedance)        : {np.mean(a)-np.mean(b):+.4f}")
    print(f"    Paired t-test p            : {t_p:.3e}  {_sig_stars(t_p)}")
    print(f"    Wilcoxon signed-rank p     : {w_p:.3e}  {_sig_stars(w_p)}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _load_policy_r_by_severity(pname: str) -> list:
    """Load per-quartile mean Pearson r for a named policy from its trials CSV."""
    path = OUT_RAW / f"{pname}_trials.csv"
    if not path.exists():
        return [float("nan")] * SEVERITY_BINS
    by_q: dict = {q: [] for q in range(SEVERITY_BINS)}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                by_q[int(row["severity_bin"])].append(float(row["pearson_r"]))
            except (KeyError, ValueError):
                pass
    return [float(np.mean(by_q[q])) if by_q[q] else float("nan")
            for q in range(SEVERITY_BINS)]


def _load_policy_overall_r(pname: str):
    """Return (mean, std) Pearson r across all episodes for a named policy."""
    path = OUT_RAW / f"{pname}_trials.csv"
    if not path.exists():
        return float("nan"), float("nan")
    vals = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                vals.append(float(row["pearson_r"]))
            except (KeyError, ValueError):
                pass
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def plot_pearsonr_bar_with_impedance(imp_trials: list) -> None:
    imp_r   = np.array([t["pearson_r"] for t in imp_trials])
    imp_mean = float(np.mean(imp_r))
    imp_std  = float(np.std(imp_r))

    ordered_policies = [
        "no_exo",
        "impedance_baseline",
        "policy_deg",
        "policy_brady_deg",
        "policy_deg_recurrent",
        "policy_brady_deg_recurrent",
    ]
    labels, means, stds, colors = [], [], [], []
    for pname in ordered_policies:
        disp, color, _ = POLICY_DISPLAY[pname]
        if pname == "impedance_baseline":
            m, s = imp_mean, imp_std
        else:
            m, s = _load_policy_overall_r(pname)
            if np.isnan(m):
                continue
        labels.append(disp); means.append(m); stds.append(s); colors.append(color)

    x   = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.6), 5))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="Healthy (r = 1.0)")
    for xi, (m, s) in enumerate(zip(means, stds)):
        if not np.isnan(m):
            ax.text(xi, m + s + 0.012, f"{m:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Mean Pearson $r$ vs Healthy")
    ax.set_title("Trajectory Recovery — Impedance Baseline vs RL Policies")
    ax.legend(); ax.set_ylim(0, 1.1)
    plt.tight_layout()
    path = OUT_PLOTS / "comparison_pearsonr_with_impedance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path.name}")


def plot_pearsonr_by_severity_with_impedance(imp_trials: list) -> None:
    imp_by_q = [
        float(np.mean([t["pearson_r"] for t in imp_trials if t["severity_bin"] == q]))
        if any(t["severity_bin"] == q for t in imp_trials) else float("nan")
        for q in range(SEVERITY_BINS)
    ]

    ordered = [
        "no_exo",
        "policy_deg_recurrent",
        "policy_brady_deg_recurrent",
        "impedance_baseline",
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    for pname in ordered:
        disp, color, ls = POLICY_DISPLAY[pname]
        if pname == "impedance_baseline":
            vals = imp_by_q
        else:
            vals = _load_policy_r_by_severity(pname)
            if all(np.isnan(v) for v in vals):
                continue
        ax.plot(range(SEVERITY_BINS), vals, ls, color=color,
                linewidth=2, label=disp, markersize=7)

    ax.set_xticks(range(SEVERITY_BINS))
    ax.set_xticklabels(SEV_LABELS)
    ax.set_ylabel("Mean Pearson $r$ vs Healthy")
    ax.set_title("Trajectory Recovery by Severity — Impedance Baseline vs RL")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = OUT_PLOTS / "pearsonr_by_severity_with_impedance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path.name}")


# ---------------------------------------------------------------------------
# Console print helpers
# ---------------------------------------------------------------------------

def print_summary(trials: list, per_q: list) -> None:
    all_r  = [t["pearson_r"]    for t in trials]
    all_g  = [t["goal_achieved"] for t in trials]
    all_rw = [t["reward"]       for t in trials]

    print(f"\n{'='*60}")
    print("Impedance Baseline — Final Summary")
    print(f"  Episodes             : {len(trials)}")
    print(f"  Overall Pearson r    : {np.mean(all_r):.4f} ± {np.std(all_r):.4f}")
    print(f"  Goal achievement     : {np.mean(all_g):.4f}  ({100*np.mean(all_g):.1f}%)")
    print(f"  Mean episode reward  : {np.mean(all_rw):.2f}")
    print()
    print(f"  {'Quartile':12s}  {'Pearson r ± std':>18}  {'Goal rate':>10}  {'Reward':>10}")
    print("  " + "-" * 58)
    for row in per_q:
        r_str = f"{row['pearson_r_mean']:.4f} ± {row['pearson_r_std']:.4f}"
        print(f"  {row['severity_quartile']:12s}  {r_str:>18}  "
              f"{row['goal_rate']:10.4f}  {row['reward_mean']:10.2f}")
    print(f"{'='*60}")


def print_latex_rows(trials: list, per_q: list, ref_r: dict) -> None:
    """Print LaTeX table rows for direct paste into the appendix."""
    all_r = [t["pearson_r"] for t in trials]
    all_g = [t["goal_achieved"] for t in trials]

    print("\n--- LaTeX rows for Table A1 (append to per-severity table) ---")
    for row in per_q:
        print(f"    Impedance & {row['severity_quartile']} & "
              f"{row['pearson_r_mean']:.3f} $\\pm$ {row['pearson_r_std']:.3f} & "
              f"— & {row['goal_rate']:.3f} \\\\")

    # Significance row
    imp_by_idx = {t["trial_idx"]: t["pearson_r"] for t in trials}
    common = sorted(set(imp_by_idx) & set(ref_r))
    if len(common) >= 2:
        a    = np.array([ref_r[i]        for i in common])
        b    = np.array([imp_by_idx[i]   for i in common])
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        _, t_p = ttest_rel(a, b)
        try:
            _, w_p = wilcoxon(a - b, alternative="two-sided")
        except ValueError:
            w_p = float("nan")
        print("\n--- LaTeX rows for Table A2 (statistical significance) ---")
        print(f"    BD RecPPO vs Impedance & Pearson $r$ & "
              f"{t_p:.2e} & {w_p:.2e} \\\\")
        g_imp = {t["trial_idx"]: t["goal_achieved"] for t in trials}
        g_ref_csv = OUT_RAW / "policy_brady_deg_recurrent_trials.csv"
        if g_ref_csv.exists():
            g_ref = {}
            with open(g_ref_csv, newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        g_ref[int(row["trial_idx"])] = float(row["goal_achieved"])
                    except (KeyError, ValueError):
                        pass
            common_g = sorted(set(g_imp) & set(g_ref))
            if len(common_g) >= 2:
                ga = np.array([g_ref[i]  for i in common_g])
                gb = np.array([g_imp[i]  for i in common_g])
                _, gt_p = ttest_rel(ga, gb)
                try:
                    _, gw_p = wilcoxon(ga - gb, alternative="two-sided")
                except ValueError:
                    gw_p = float("nan")
                print(f"    BD RecPPO vs Impedance & Goal rate & "
                      f"{gt_p:.2e} & {gw_p:.2e} \\\\")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ReGainX impedance baseline evaluation")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Limit to first N episodes (0 = all)")
    args = parser.parse_args()

    for _d in [OUT_RAW, OUT_SUMMARIES, OUT_PLOTS]:
        _d.mkdir(parents=True, exist_ok=True)

    if not CONFIGS_CSV.exists():
        raise FileNotFoundError(
            f"Episode configs not found: {CONFIGS_CSV}\n"
            "Run: python run_shared_evaluation.py --episodes 4000"
        )
    if not HEALTHY_PATH.exists():
        raise FileNotFoundError(f"Healthy policy not found: {HEALTHY_PATH}")

    print(f"\n{'='*60}")
    print("ReGainX — Impedance-Control Baseline Evaluation")
    print(f"  Configs  : {CONFIGS_CSV}")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*60}\n")

    configs = load_configs(CONFIGS_CSV)
    if args.episodes > 0:
        configs = configs[: args.episodes]
        print(f"  [limited to {args.episodes} episodes]")
    print(f"  Loaded {len(configs)} episode configs")

    # Healthy reference trajectories
    print("\n[Healthy Reference]")
    healthy_results = run_healthy_reference(configs)

    # Impedance evaluation
    print("\n[Impedance Evaluation]")
    trials = run_impedance_eval(configs, healthy_results)

    # Summaries
    per_q = save_per_quartile(trials)
    per_r = save_per_radian(trials)
    print_summary(trials, per_q)

    # Significance vs BD RecPPO
    print("\n[Statistical Significance]")
    ref_r = load_csv_r(RECPPO_CSV)
    if ref_r:
        print_significance(trials, ref_r, "BD RecPPO")
    else:
        print("  BD RecPPO CSV not found — run run_shared_evaluation.py first")

    # LaTeX rows
    print_latex_rows(trials, per_q, ref_r)

    # Plots
    print("\n[Plots]")
    plot_pearsonr_bar_with_impedance(trials)
    plot_pearsonr_by_severity_with_impedance(trials)

    print(f"\nAll outputs saved to {OUT_DIR}\n")


if __name__ == "__main__":
    main()
