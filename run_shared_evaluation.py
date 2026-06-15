"""
run_shared_evaluation.py  --  ReGainX Shared-Episode Policy Evaluation
=======================================================================

All policies are evaluated on identical episode configurations so that
per-episode results are directly comparable across policies.

  Phase 1  Generate N episode configs.  Start angle comes from MyoSuite's
           own reset (matching the training distribution); target angle is
           sampled from the joint range.
  Phase 2  Run the healthy reference policy ONCE per episode.
  Phase 3  Run each exo / ablation policy on the same N episodes.

Stratification axes
  Severity quartiles  Q1 (mild) → Q4 (severe) via force_scale +
                      activation_slowdown + avg_mf composite.
  Radian travelled    |target_angle - start_angle| in 4 fixed bins:
                      0–0.5, 0.5–1.0, 1.0–1.5, 1.5+ rad.

Usage
-----
  python run_shared_evaluation.py                   # 4 000-episode full run
  python run_shared_evaluation.py --episodes 400    # quick smoke test
  python run_shared_evaluation.py --skip-existing   # skip policies with CSV

Output
------
  results/shared_eval/
    episode_configs.csv               shared configs (one row per episode)
    raw/  {policy}_trials.csv         one row per episode per policy
    summaries/
      all_policies_summary.csv
      {policy}_per_quartile.csv
      {policy}_per_radian.csv
      {policy}_per_cell.csv           4x4  radian_bin x severity_bin
    plots/
      comparison_reward_bar.png
      comparison_pearsonr_bar.png
      reward_by_severity.png
      reward_by_radian.png
      pearsonr_by_severity.png
      pearsonr_by_radian.png
      {policy}_heatmap.png
      {policy}_reward_heatmap.png
"""

import argparse
import csv
import os
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import myosuite  # noqa: F401 — registers MyoSuite envs
from myosuite.utils import gym
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECPPO = True
except ImportError:
    _HAS_RECPPO = False
    warnings.warn("sb3_contrib not installed — RecurrentPPO policies will be skipped.")

from envs.elbow_env import CombinedExoOnlyWrapper
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

for _d in [OUT_RAW, OUT_SUMMARIES, OUT_PLOTS]:
    _d.mkdir(parents=True, exist_ok=True)

HEALTHY_PATH = POLICIES_DIR / "healthy_policy.zip"

SEVERITY_BINS = 4
RADIAN_BINS   = 4
SEV_LABELS    = ["Q1 mild", "Q2", "Q3", "Q4 severe"]
RADIAN_EDGES  = [0.0, 0.5, 1.0, 1.5, float("inf")]
RADIAN_LABELS = ["0.0–0.5 rad", "0.5–1.0 rad", "1.0–1.5 rad", "1.5+ rad"]
MAX_STEPS     = 200

TRIAL_FIELDS = [
    "trial_idx", "radian_bin", "severity_bin",
    "start_angle", "target_angle", "radian_travelled",
    "force_scale", "activation_slowdown", "avg_mf", "severity",
    "pearson_r", "reward", "goal_achieved", "goal_time_s",
    "episode_steps", "healthy_reward",
]

# Registry: (display_name, zip_filename_or_None, is_recurrent)
POLICY_REGISTRY = [
    ("policy_brady_deg_recurrent",       "policy_brady_deg_recurrent.zip",       True),
    ("policy_deg_recurrent",             "policy_deg_recurrent.zip",             True),
    ("policy_brady_deg",                 "policy_brady_deg.zip",                 False),
    ("policy_deg",                       "policy_deg.zip",                       False),
    ("policy_brady_deg_recurrent_noisy", "policy_brady_deg_recurrent_noisy.zip", True),
    ("no_exo",                           None,                                   False),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def severity_quartile_to_range(q):
    fs_e = np.linspace(0.6, 0.9, 5)[::-1]
    sl_e = np.linspace(1.1, 1.4, 5)
    mf_e = np.linspace(0.0, 1.0, 5)
    return (
        (float(fs_e[q + 1]), float(fs_e[q])),
        (float(sl_e[q]),     float(sl_e[q + 1])),
        (float(mf_e[q]),     float(mf_e[q + 1])),
    )


def to_radian_bin(rad):
    for i in range(RADIAN_BINS):
        if rad < RADIAN_EDGES[i + 1]:
            return i
    return RADIAN_BINS - 1


def _safe_r(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    r, _ = pearsonr(a[:n], b[:n])
    return float(r) if not np.isnan(r) else 0.0


def _flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def make_exo_env():
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    return CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=str(HEALTHY_PATH),
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )


def load_policy(path, is_recurrent):
    if is_recurrent:
        if not _HAS_RECPPO:
            raise ImportError("sb3_contrib required for RecurrentPPO")
        return RecurrentPPO.load(path)
    return PPO.load(path)


# ---------------------------------------------------------------------------
# Environment configuration for episode replay
# ---------------------------------------------------------------------------

def configure_exo_for_replay(base_env, cfg):
    """Set exo env to exactly the saved episode config. Start angle is NOT zeroed."""
    buw = base_env.base_env.unwrapped

    buw.target_jnt_value = [cfg["target_angle"]]
    buw.target_type      = "fixed"
    buw.update_target(restore_sim=True)

    mf, sp = cfg["mf_vals"], cfg["split_vals"]
    rem = 1.0 - mf
    buw.muscle_fatigue.MA[:] = rem * sp
    buw.muscle_fatigue.MR[:] = rem * (1.0 - sp)
    buw.muscle_fatigue.MF[:] = mf

    base_env.force_scale         = cfg["force_scale"]
    base_env.activation_slowdown = cfg["activation_slowdown"]
    base_env._apply_brady()

    buw.sim.data.qpos[0] = cfg["start_angle"]
    buw.sim.data.qvel[:] = 0.0
    buw.sim.forward()

    return base_env._build_obs(base_env._current_raw_obs())


def configure_healthy_for_replay(healthy_env, cfg):
    """Set healthy env to exactly the saved episode config."""
    u = healthy_env.unwrapped
    u.target_jnt_value = [cfg["target_angle"]]
    u.target_type      = "fixed"
    u.update_target(restore_sim=True)
    u.sim.data.qpos[0] = cfg["start_angle"]
    u.sim.data.qvel[:] = 0.0
    u.sim.forward()


# ---------------------------------------------------------------------------
# Phase 1: Generate shared episode configs
# ---------------------------------------------------------------------------

def generate_configs(n_episodes, seed=42):
    """
    Generate episode configs.  Start angle is drawn from MyoSuite's reset so
    it matches the training distribution.  Configs are distributed evenly
    across severity bins; radian_bin is computed from the resulting
    |target - start| distance.
    """
    np.random.seed(seed)

    exo_env   = make_exo_env()
    n_muscles = exo_env.n_muscles

    tmp      = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    tgt_low  = float(tmp.unwrapped.target_jnt_range[0, 0])
    tgt_high = float(tmp.unwrapped.target_jnt_range[0, 1])
    tmp.close()

    # Distribute evenly across severity bins (shuffle for randomness)
    base_n   = n_episodes // SEVERITY_BINS
    sev_plan = []
    for q in range(SEVERITY_BINS):
        cnt = base_n + (1 if q < n_episodes % SEVERITY_BINS else 0)
        sev_plan.extend([q] * cnt)
    np.random.shuffle(sev_plan)

    configs = []
    for i, sev_q in enumerate(sev_plan):
        # Let MyoSuite randomize the start position (matches training)
        exo_env.reset()
        start_angle = float(exo_env.base_env.unwrapped.sim.data.qpos[0])

        target_angle   = float(np.random.uniform(tgt_low, tgt_high))
        rad_travelled  = abs(target_angle - start_angle)
        rb             = to_radian_bin(rad_travelled)

        fs_r, sl_r, mf_r = severity_quartile_to_range(sev_q)
        force_scale = float(np.random.uniform(*fs_r))
        act_slow    = float(np.random.uniform(*sl_r))
        avg_mf_t    = float(np.random.uniform(*mf_r))
        mf_vals     = np.random.uniform(
            max(avg_mf_t * 0.9, 0.0), min(avg_mf_t * 1.1, 1.0), size=n_muscles)
        split_vals  = np.random.uniform(0.0, 1.0, size=n_muscles)
        severity    = compute_severity(force_scale, act_slow, float(np.mean(mf_vals)))

        configs.append({
            "trial_idx":          i,
            "severity_bin":       sev_q,
            "radian_bin":         rb,
            "start_angle":        start_angle,
            "target_angle":       target_angle,
            "radian_travelled":   rad_travelled,
            "force_scale":        force_scale,
            "activation_slowdown": act_slow,
            "avg_mf":             float(np.mean(mf_vals)),
            "severity":           severity,
            "mf_vals":            mf_vals,
            "split_vals":         split_vals,
            "healthy_reward":     float("nan"),
        })

    exo_env.close()
    print(f"  Generated {len(configs)} episode configs  "
          f"(target range [{tgt_low:.2f}, {tgt_high:.2f}] rad)")
    for b in range(RADIAN_BINS):
        cnt = sum(1 for c in configs if c["radian_bin"] == b)
        print(f"    {RADIAN_LABELS[b]:15s}: {cnt:4d} episodes ({100*cnt/len(configs):.1f}%)")

    return configs


def save_configs_csv(configs, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "trial_idx", "severity_bin", "radian_bin",
            "start_angle", "target_angle", "radian_travelled",
            "force_scale", "activation_slowdown", "avg_mf", "severity",
            "healthy_reward", "mf_vals", "split_vals",
        ])
        for c in configs:
            w.writerow([
                c["trial_idx"], c["severity_bin"], c["radian_bin"],
                f"{c['start_angle']:.6f}", f"{c['target_angle']:.6f}",
                f"{c['radian_travelled']:.6f}", f"{c['force_scale']:.6f}",
                f"{c['activation_slowdown']:.6f}", f"{c['avg_mf']:.6f}",
                f"{c['severity']:.6f}",
                f"{c['healthy_reward']:.6f}" if not np.isnan(c["healthy_reward"]) else "",
                ";".join(f"{v:.6f}" for v in c["mf_vals"]),
                ";".join(f"{v:.6f}" for v in c["split_vals"]),
            ])


# ---------------------------------------------------------------------------
# Phase 2: Run healthy policy on all configs (once)
# ---------------------------------------------------------------------------

def run_healthy_phase(configs):
    """Run healthy reference policy on every config. Fills configs[i]['healthy_reward']."""
    healthy_env    = gym.make("myoElbowPose1D6MRandom-v0")
    healthy_policy = PPO.load(str(HEALTHY_PATH))
    obs_dim        = healthy_policy.observation_space.shape[0]

    results = []
    t0      = time.time()
    print(f"  Running healthy reference on {len(configs)} episodes...")

    for i, cfg in enumerate(configs):
        healthy_env.reset()
        configure_healthy_for_replay(healthy_env, cfg)

        obs          = healthy_env.unwrapped.get_obs()[:obs_dim]
        angles       = []
        total_reward = 0.0
        goal_achieved, goal_time = False, float("nan")

        for step in range(MAX_STEPS):
            action, _ = healthy_policy.predict(obs, deterministic=True)
            next_obs, rwd, done, truncated, _ = healthy_env.step(action)
            obs = next_obs[:obs_dim]
            total_reward += float(rwd)
            angles.append(float(healthy_env.unwrapped.sim.data.qpos[0]))
            solved = healthy_env.unwrapped.rwd_dict.get("solved", False)
            if bool(np.asarray(solved).flat[0]) and not goal_achieved:
                goal_achieved = True
                goal_time     = step * healthy_env.unwrapped.dt
            if done or truncated:
                break

        configs[i]["healthy_reward"] = total_reward
        results.append({
            "angles":        angles,
            "reward":        total_reward,
            "goal_achieved": int(goal_achieved),
            "goal_time":     goal_time,
            "steps":         len(angles),
        })

        if (i + 1) % 500 == 0 or (i + 1) == len(configs):
            elapsed = time.time() - t0
            print(f"    [{i+1:4d}/{len(configs)}]  elapsed={elapsed:.0f}s")

    healthy_env.close()
    return results


# ---------------------------------------------------------------------------
# Phase 3: Run one policy on all shared configs
# ---------------------------------------------------------------------------

def run_policy_phase(policy_name, policy_path, is_recurrent,
                     configs, healthy_results, skip_if_exists=False):
    csv_path = OUT_RAW / f"{policy_name}_trials.csv"
    if skip_if_exists and csv_path.exists():
        print(f"  [skip] {policy_name} — CSV exists")
        return _load_csv(csv_path)

    is_noexo = (policy_path is None)
    policy   = None if is_noexo else load_policy(policy_path, is_recurrent)

    exo_env  = make_exo_env()
    base_env = exo_env
    zero     = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    trials   = []
    t0       = time.time()

    print(f"\n  Evaluating: {policy_name}  (recurrent={is_recurrent})")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_FIELDS)
        writer.writeheader()

        for i, (cfg, h_res) in enumerate(zip(configs, healthy_results)):
            exo_env.reset()
            obs = configure_exo_for_replay(base_env, cfg)

            angles        = []
            total_reward  = 0.0
            goal_achieved = False
            goal_time     = float("nan")
            lstm_states   = None
            ep_start      = np.ones((1,), dtype=bool)

            for step in range(MAX_STEPS):
                if is_noexo:
                    action = zero
                elif is_recurrent:
                    action, lstm_states = policy.predict(
                        obs, state=lstm_states,
                        episode_start=ep_start, deterministic=True)
                    ep_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = policy.predict(obs, deterministic=True)

                obs, rwd, done, truncated, _ = exo_env.step(action)
                total_reward += float(rwd)
                angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))

                solved = base_env.base_env.unwrapped.rwd_dict.get("solved", False)
                if bool(np.asarray(solved).flat[0]) and not goal_achieved:
                    goal_achieved = True
                    goal_time     = step * base_env.base_env.unwrapped.dt
                if done or truncated:
                    break

            pr  = _safe_r(angles, h_res["angles"])
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
                "goal_time_s": (f"{goal_time:.6f}"
                                if not np.isnan(float(goal_time)) else ""),
                "episode_steps":       len(angles),
                "healthy_reward":      f"{cfg['healthy_reward']:.6f}",
            }
            trials.append(row)
            writer.writerow(row)
            f.flush()

            if (i + 1) % 500 == 0 or (i + 1) == len(configs):
                elapsed = time.time() - t0
                eta     = elapsed / (i + 1) * (len(configs) - i - 1)
                print(f"    [{i+1:4d}/{len(configs)}]  r={pr:.3f}  "
                      f"goal={int(goal_achieved)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    exo_env.close()
    print(f"  Saved → {csv_path}")
    return trials


def _load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def to_arrays(trials):
    keys = ["pearson_r", "reward", "goal_achieved", "episode_steps",
            "radian_bin", "severity_bin", "radian_travelled", "healthy_reward"]
    return {k: np.array([_flt(t.get(k, "")) for t in trials]) for k in keys}


def _pct(arr):
    arr = arr[~np.isnan(arr)]
    if not len(arr):
        return {k: float("nan") for k in
                ["mean", "std", "min", "max", "p5", "p50", "p95", "n"]}
    return {
        "mean": float(np.mean(arr)),  "std": float(np.std(arr)),
        "min":  float(np.min(arr)),   "max": float(np.max(arr)),
        "p5":   float(np.percentile(arr,  5)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "n":    len(arr),
    }


def compute_summary(policy_name, trials):
    arrs  = to_arrays(trials)
    stats = {"policy": policy_name, "n_episodes": len(trials)}
    for metric in ["pearson_r", "reward", "goal_achieved", "episode_steps"]:
        for k, v in _pct(arrs[metric]).items():
            stats[f"{metric}_{k}"] = round(v, 6) if isinstance(v, float) else v
    for q in range(SEVERITY_BINS):
        m = arrs["severity_bin"] == q
        pr = arrs["pearson_r"][m]; pr = pr[~np.isnan(pr)]
        rw = arrs["reward"][m];    rw = rw[~np.isnan(rw)]
        g  = arrs["goal_achieved"][m]
        stats[f"pearson_r_Q{q+1}_mean"] = round(float(np.mean(pr)), 6) if len(pr) else float("nan")
        stats[f"pearson_r_Q{q+1}_std"]  = round(float(np.std(pr)),  6) if len(pr) else float("nan")
        stats[f"reward_Q{q+1}_mean"]    = round(float(np.mean(rw)), 6) if len(rw) else float("nan")
        stats[f"goal_rate_Q{q+1}"]      = round(float(np.mean(g)),  6) if len(g)  else float("nan")
    for b in range(RADIAN_BINS):
        m = arrs["radian_bin"] == b
        pr = arrs["pearson_r"][m]; pr = pr[~np.isnan(pr)]
        rw = arrs["reward"][m];    rw = rw[~np.isnan(rw)]
        g  = arrs["goal_achieved"][m]
        stats[f"pearson_r_R{b+1}_mean"] = round(float(np.mean(pr)), 6) if len(pr) else float("nan")
        stats[f"pearson_r_R{b+1}_std"]  = round(float(np.std(pr)),  6) if len(pr) else float("nan")
        stats[f"reward_R{b+1}_mean"]    = round(float(np.mean(rw)), 6) if len(rw) else float("nan")
        stats[f"goal_rate_R{b+1}"]      = round(float(np.mean(g)),  6) if len(g)  else float("nan")
    return stats


def save_per_quartile(policy_name, trials):
    arrs, rows = to_arrays(trials), []
    for q in range(SEVERITY_BINS):
        m   = arrs["severity_bin"] == q
        pr  = arrs["pearson_r"][m]; pr = pr[~np.isnan(pr)]
        rw  = arrs["reward"][m];    rw = rw[~np.isnan(rw)]
        g   = arrs["goal_achieved"][m]
        s   = _pct(pr)
        row = {"severity_quartile": SEV_LABELS[q]}
        row.update({f"pearson_r_{k}": round(v, 6) if isinstance(v, float) else v
                    for k, v in s.items()})
        row["goal_rate"]   = round(float(np.mean(g)), 6) if len(g) else float("nan")
        row["reward_mean"] = round(float(np.mean(rw)), 6) if len(rw) else float("nan")
        row["reward_std"]  = round(float(np.std(rw)),  6) if len(rw) else float("nan")
        rows.append(row)
    _write_csv(OUT_SUMMARIES / f"{policy_name}_per_quartile.csv", rows)


def save_per_radian(policy_name, trials):
    arrs, rows = to_arrays(trials), []
    for b in range(RADIAN_BINS):
        m   = arrs["radian_bin"] == b
        pr  = arrs["pearson_r"][m]; pr = pr[~np.isnan(pr)]
        rw  = arrs["reward"][m];    rw = rw[~np.isnan(rw)]
        g   = arrs["goal_achieved"][m]
        s   = _pct(pr)
        row = {"radian_bin": RADIAN_LABELS[b]}
        row.update({f"pearson_r_{k}": round(v, 6) if isinstance(v, float) else v
                    for k, v in s.items()})
        row["goal_rate"]   = round(float(np.mean(g)), 6) if len(g) else float("nan")
        row["reward_mean"] = round(float(np.mean(rw)), 6) if len(rw) else float("nan")
        row["reward_std"]  = round(float(np.std(rw)),  6) if len(rw) else float("nan")
        rows.append(row)
    _write_csv(OUT_SUMMARIES / f"{policy_name}_per_radian.csv", rows)


def save_per_cell(policy_name, trials):
    arrs, rows = to_arrays(trials), []
    for b in range(RADIAN_BINS):
        for q in range(SEVERITY_BINS):
            m   = (arrs["radian_bin"] == b) & (arrs["severity_bin"] == q)
            pr  = arrs["pearson_r"][m]; pr = pr[~np.isnan(pr)]
            rw  = arrs["reward"][m];    rw = rw[~np.isnan(rw)]
            g   = arrs["goal_achieved"][m]
            s   = _pct(pr)
            rows.append({
                "radian_bin":     RADIAN_LABELS[b],
                "severity_bin":   SEV_LABELS[q],
                "n":              s["n"],
                "pearson_r_mean": round(s["mean"], 6),
                "pearson_r_std":  round(s["std"],  6),
                "pearson_r_p50":  round(s["p50"],  6),
                "goal_rate":      round(float(np.mean(g)), 6) if len(g) else float("nan"),
                "reward_mean":    round(float(np.mean(rw)), 6) if len(rw) else float("nan"),
                "reward_std":     round(float(np.std(rw)),  6) if len(rw) else float("nan"),
            })
    _write_csv(OUT_SUMMARIES / f"{policy_name}_per_cell.csv", rows)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _palette(names):
    cmap = plt.cm.tab10(np.linspace(0, 0.9, max(len(names), 1)))
    return dict(zip(names, cmap))


def plot_bar(all_trials, configs, metric, ylabel, title, fname, healthy_label="Healthy reference"):
    """Bar chart of a metric per policy, with healthy reference dashed line."""
    names  = list(all_trials.keys())
    colors = _palette(names)
    means, stds = [], []
    for trials in all_trials.values():
        vals = to_arrays(trials)[metric]
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals)))

    # Healthy reference from configs (always reward)
    h_vals = [c["healthy_reward"] for c in configs if not np.isnan(c["healthy_reward"])]
    h_mean = float(np.nanmean(h_vals)) if h_vals else float("nan")

    x   = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.8), 6))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[n] for n in names], alpha=0.85)

    if metric == "reward" and not np.isnan(h_mean):
        ax.axhline(h_mean, color="green", linestyle="--", linewidth=1.8,
                   label=f"{healthy_label} ({h_mean:.1f})")
        ax.legend()

    for bar, m in zip(bars, means):
        if not np.isnan(m):
            offset = max(abs(m) * 0.02, 1.0)
            ax.text(bar.get_x() + bar.get_width() / 2, m + offset,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {fname}")


def plot_by_bin(all_trials, configs, bin_key, bin_labels, metric, ylabel, title, fname):
    """Line plot of metric across stratification bins, one line per policy."""
    n_bins = len(bin_labels)
    names  = list(all_trials.keys())
    colors = _palette(names)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Healthy reference line (only for reward)
    if metric == "reward":
        h_means = []
        for b in range(n_bins):
            vals = [c["healthy_reward"] for c in configs
                    if c[bin_key] == b and not np.isnan(c["healthy_reward"])]
            h_means.append(float(np.mean(vals)) if vals else float("nan"))
        ax.plot(range(n_bins), h_means, "k--", linewidth=1.8,
                label="Healthy reference", zorder=5)

    for pname, trials in all_trials.items():
        arrs        = to_arrays(trials)
        means, lo, hi = [], [], []
        for b in range(n_bins):
            mask = arrs[bin_key] == b
            vals = arrs[metric][mask]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                m, s = float(np.mean(vals)), float(np.std(vals))
                means.append(m); lo.append(m - s); hi.append(m + s)
            else:
                means.append(float("nan"))
                lo.append(float("nan")); hi.append(float("nan"))

        color = colors[pname]
        ax.plot(range(n_bins), means, "o-", color=color, linewidth=2, label=pname)
        ax.fill_between(range(n_bins), lo, hi, color=color, alpha=0.12)

    ax.set_xticks(range(n_bins))
    ax.set_xticklabels(bin_labels, rotation=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {fname}")


def plot_heatmap(policy_name, trials, metric="pearson_r"):
    arrs   = to_arrays(trials)
    matrix = np.full((RADIAN_BINS, SEVERITY_BINS), np.nan)
    for rb in range(RADIAN_BINS):
        for sq in range(SEVERITY_BINS):
            mask = (arrs["radian_bin"] == rb) & (arrs["severity_bin"] == sq)
            vals = arrs[metric][mask]; vals = vals[~np.isnan(vals)]
            if len(vals):
                matrix[rb, sq] = float(np.mean(vals))

    if metric == "pearson_r":
        vmin, vmax, cmap = 0.0, 1.0, "Blues"
        cbar_label = "Mean Pearson r vs Healthy"
    else:
        finite = matrix[np.isfinite(matrix)]
        vmin   = float(np.nanmin(finite)) if len(finite) else 0.0
        vmax   = float(np.nanmax(finite)) if len(finite) else 1.0
        cmap   = "viridis"
        cbar_label = "Mean Episode Reward"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(SEVERITY_BINS)); ax.set_xticklabels(SEV_LABELS)
    ax.set_yticks(range(RADIAN_BINS));   ax.set_yticklabels(RADIAN_LABELS)
    ax.set_xlabel("Severity Quartile")
    ax.set_ylabel("Radian Travelled")
    ax.set_title(f"{policy_name}  —  {cbar_label}")
    plt.colorbar(im, ax=ax, label=cbar_label)
    for i in range(RADIAN_BINS):
        for j in range(SEVERITY_BINS):
            v = matrix[i, j]
            txt   = f"{v:.3f}" if not np.isnan(v) else "N/A"
            color = "white" if (not np.isnan(v) and v > vmin + (vmax - vmin) * 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    plt.tight_layout()
    tag  = "heatmap" if metric == "pearson_r" else "reward_heatmap"
    path = OUT_PLOTS / f"{policy_name}_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {path.name}")


def plot_reward_comparison_panel(all_trials, configs):
    """3-panel figure: overall bar + by severity + by radian (reward only)."""
    names  = list(all_trials.keys())
    colors = _palette(names)
    h_vals = [c["healthy_reward"] for c in configs if not np.isnan(c["healthy_reward"])]
    h_mean = float(np.nanmean(h_vals)) if h_vals else float("nan")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Reward Comparison — All Policies (Shared Episode Configs)", fontsize=13)

    # Panel 1: Overall bar
    ax = axes[0]
    means = [float(np.nanmean(to_arrays(t)["reward"])) for t in all_trials.values()]
    stds  = [float(np.nanstd(to_arrays(t)["reward"]))  for t in all_trials.values()]
    x     = np.arange(len(names))
    bars  = ax.bar(x, means, yerr=stds, capsize=4,
                   color=[colors[n] for n in names], alpha=0.85)
    if not np.isnan(h_mean):
        ax.axhline(h_mean, color="green", linestyle="--", linewidth=1.5,
                   label=f"Healthy ({h_mean:.1f})")
        ax.legend(fontsize=7)
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + max(abs(m) * 0.02, 1.0),
                    f"{m:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Mean Episode Reward"); ax.set_title("Overall")

    # Panel 2: By severity
    ax = axes[1]
    h_sev = []
    for q in range(SEVERITY_BINS):
        v = [c["healthy_reward"] for c in configs
             if c["severity_bin"] == q and not np.isnan(c["healthy_reward"])]
        h_sev.append(float(np.mean(v)) if v else float("nan"))
    ax.plot(range(SEVERITY_BINS), h_sev, "k--", linewidth=1.5,
            label="Healthy", zorder=5)
    for pname, trials in all_trials.items():
        arrs = to_arrays(trials)
        ms   = [float(np.nanmean(arrs["reward"][arrs["severity_bin"] == q]))
                for q in range(SEVERITY_BINS)]
        ax.plot(range(SEVERITY_BINS), ms, "o-", color=colors[pname],
                linewidth=2, label=pname)
    ax.set_xticks(range(SEVERITY_BINS)); ax.set_xticklabels(SEV_LABELS, rotation=10, fontsize=7)
    ax.set_ylabel("Mean Reward"); ax.set_title("By Severity Quartile")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    # Panel 3: By radian travelled
    ax = axes[2]
    h_rad = []
    for b in range(RADIAN_BINS):
        v = [c["healthy_reward"] for c in configs
             if c["radian_bin"] == b and not np.isnan(c["healthy_reward"])]
        h_rad.append(float(np.mean(v)) if v else float("nan"))
    ax.plot(range(RADIAN_BINS), h_rad, "k--", linewidth=1.5,
            label="Healthy", zorder=5)
    for pname, trials in all_trials.items():
        arrs = to_arrays(trials)
        ms   = [float(np.nanmean(arrs["reward"][arrs["radian_bin"] == b]))
                for b in range(RADIAN_BINS)]
        ax.plot(range(RADIAN_BINS), ms, "s-", color=colors[pname],
                linewidth=2, label=pname)
    ax.set_xticks(range(RADIAN_BINS)); ax.set_xticklabels(RADIAN_LABELS, rotation=10, fontsize=7)
    ax.set_ylabel("Mean Reward"); ax.set_title("By Radian Travelled")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "reward_comparison_panel.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot → reward_comparison_panel.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ReGainX shared-episode policy evaluation")
    parser.add_argument("--episodes",      type=int, default=4000)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip per-policy evaluation if raw CSV already exists")
    args = parser.parse_args()

    N = args.episodes
    print(f"\n{'='*60}")
    print(f"ReGainX Shared-Episode Evaluation")
    print(f"  Episodes : {N}  |  Seed: {args.seed}")
    print(f"  Output   : {OUT_DIR}")
    print(f"{'='*60}")

    # ---- Phase 1 -----------------------------------------------------------
    print(f"\n[Phase 1] Generating {N} shared episode configs...")
    configs = generate_configs(N, seed=args.seed)

    # ---- Phase 2 -----------------------------------------------------------
    print(f"\n[Phase 2] Running healthy reference policy (once per episode)...")
    healthy_results = run_healthy_phase(configs)
    save_configs_csv(configs, OUT_DIR / "episode_configs.csv")
    print(f"  Configs saved → episode_configs.csv")

    # ---- Phase 3 -----------------------------------------------------------
    print(f"\n[Phase 3] Evaluating policies on shared configs...")
    all_trials = {}
    all_stats  = {}

    for pname, pfile, is_rec in POLICY_REGISTRY:
        if pfile is not None and not (POLICIES_DIR / pfile).exists():
            print(f"  [skip] {pname} — zip not found")
            continue
        ppath  = str(POLICIES_DIR / pfile) if pfile else None
        trials = run_policy_phase(
            pname, ppath, is_rec,
            configs, healthy_results,
            skip_if_exists=args.skip_existing,
        )
        all_trials[pname] = trials
        all_stats[pname]  = compute_summary(pname, trials)
        save_per_quartile(pname, trials)
        save_per_radian(pname, trials)
        save_per_cell(pname, trials)
        plot_heatmap(pname, trials, "pearson_r")
        plot_heatmap(pname, trials, "reward")

    # ---- Master summary CSV ------------------------------------------------
    if all_stats:
        fieldnames = sorted({k for s in all_stats.values() for k in s})
        path = OUT_SUMMARIES / "all_policies_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for s in all_stats.values():
                w.writerow(s)
        print(f"\n  Master summary → {path.name}")

    # ---- Plots -------------------------------------------------------------
    if all_trials:
        print(f"\n[Plots]")
        plot_bar(
            all_trials, configs, "reward",
            "Mean Episode Reward",
            "Reward Comparison — All Policies (Shared Episode Configs)",
            "comparison_reward_bar.png",
        )
        plot_bar(
            all_trials, configs, "pearson_r",
            "Mean Pearson r vs Healthy",
            "Trajectory Recovery — All Policies (Shared Episode Configs)",
            "comparison_pearsonr_bar.png",
        )
        plot_by_bin(
            all_trials, configs, "severity_bin", SEV_LABELS,
            "reward", "Mean Episode Reward", "Reward by Severity Quartile",
            "reward_by_severity.png",
        )
        plot_by_bin(
            all_trials, configs, "radian_bin", RADIAN_LABELS,
            "reward", "Mean Episode Reward", "Reward by Radian Travelled",
            "reward_by_radian.png",
        )
        plot_by_bin(
            all_trials, configs, "severity_bin", SEV_LABELS,
            "pearson_r", "Mean Pearson r", "Pearson r by Severity Quartile",
            "pearsonr_by_severity.png",
        )
        plot_by_bin(
            all_trials, configs, "radian_bin", RADIAN_LABELS,
            "pearson_r", "Mean Pearson r", "Pearson r by Radian Travelled",
            "pearsonr_by_radian.png",
        )
        plot_reward_comparison_panel(all_trials, configs)

    # ---- Final summary table -----------------------------------------------
    if all_stats:
        print(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
        print(f"  {'Policy':45s}  {'Pearson r':>10}  {'Goal rate':>10}  "
              f"{'Reward':>10}  {'N':>6}")
        print("  " + "-" * 90)
        for pname, s in all_stats.items():
            pr = s.get("pearson_r_mean",     float("nan"))
            gr = s.get("goal_achieved_mean", float("nan"))
            rw = s.get("reward_mean",        float("nan"))
            n  = s.get("n_episodes",         0)
            print(f"  {pname:45s}  {pr:10.4f}  {gr:10.4f}  {rw:10.2f}  {n:6d}")

    print(f"\n{'='*60}")
    print(f"Done. Results → {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
