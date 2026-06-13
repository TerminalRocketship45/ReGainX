"""
run_full_evaluation.py  --  ReGainX Comprehensive Policy Evaluation
====================================================================
Runs Steps 1-4 from the research evaluation protocol.

Usage
-----
  python run_full_evaluation.py                      # full 4000-episode run
  python run_full_evaluation.py --episodes 400       # quick smoke test (25/cell)
  python run_full_evaluation.py --skip-existing      # skip policies whose CSV exists
  python run_full_evaluation.py --step 4             # training curves only
  python run_full_evaluation.py --step 2             # evaluations only

Output
------
  results/full_eval/
    raw/
      {policy}_trials.csv          -- one row per episode
    summaries/
      all_policies_summary.csv     -- top-level table (mean/std/pct)
      {policy}_per_cell.csv        -- 4x4 heatmap values
      {policy}_per_quartile.csv    -- Q1-Q4 breakdown
      {policy}_per_angle.csv       -- angle-bin breakdown
    plots/
      {policy}_heatmap.png
      comparison_pearsonr_bar.png
      ablation_*.png
      noise_ablation.png
    training_curves/
      training_curve_stats.csv
      reward_curves_overlay.png

Metrics per episode
-------------------
  pearson_r            Pearson r (exo traj vs healthy reference)
  reward               Cumulative episode reward
  goal_achieved        1 if rwd_dict["solved"] fired, else 0
  goal_time_s          Seconds to first goal (blank = not achieved)
  episode_steps        Steps until done/truncated
  severity             Composite impairment score in [0, 1]
  force_scale          Brady force scale
  activation_slowdown  Brady slowdown factor
  avg_mf               Mean muscle fatigue
  target_angle         Target elbow angle (rad)
  angle_bin            0-3 into 4 angle bins
  severity_bin         0-3 into 4 severity quartiles
"""

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import myosuite  # noqa: F401 -- registers MyoSuite envs
from myosuite.utils import gym
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECPPO = True
except ImportError:
    _HAS_RECPPO = False
    warnings.warn("sb3_contrib not installed -- RecurrentPPO policies will be skipped.")

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.elbow_env_noisy import NoisyExoWrapper
from utils import compute_severity

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).parent
POLICIES_DIR  = ROOT / "policies"
LOGS_DIR      = ROOT / "logs"
OUT_DIR       = ROOT / "results" / "full_eval"
OUT_RAW       = OUT_DIR / "raw"
OUT_SUMMARIES = OUT_DIR / "summaries"
OUT_PLOTS     = OUT_DIR / "plots"
OUT_CURVES    = OUT_DIR / "training_curves"

for _d in [OUT_RAW, OUT_SUMMARIES, OUT_PLOTS, OUT_CURVES]:
    _d.mkdir(parents=True, exist_ok=True)

HEALTHY_PATH = POLICIES_DIR / "healthy_policy.zip"

# ---------------------------------------------------------------------------
# Evaluation grid constants
# ---------------------------------------------------------------------------
ANGLE_BINS    = 4
SEVERITY_BINS = 4
ANGLE_LABELS  = ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5"]
SEV_LABELS    = ["Q1 mild", "Q2", "Q3", "Q4 severe"]
MAX_STEPS     = 200   # safety guard (MyoSuite internally truncates at ~100)

TRIAL_CSV_HEADER = [
    "trial_idx", "angle_bin", "severity_bin", "target_angle",
    "force_scale", "activation_slowdown", "avg_mf", "severity",
    "pearson_r", "reward", "goal_achieved", "goal_time_s", "episode_steps",
]


def severity_quartile_to_range(q):
    """Map quartile index 0-3 to (force_scale, slowdown, mf) ranges."""
    fs_edges = np.linspace(0.6, 0.9, 5)[::-1]
    sl_edges = np.linspace(1.1, 1.4, 5)
    mf_edges = np.linspace(0.0, 1.0, 5)
    return (
        (float(fs_edges[q + 1]), float(fs_edges[q])),
        (float(sl_edges[q]),     float(sl_edges[q + 1])),
        (float(mf_edges[q]),     float(mf_edges[q + 1])),
    )


def plan_trials(n_total):
    """Even distribution of trials across 16 cells."""
    base = n_total // (ANGLE_BINS * SEVERITY_BINS)
    rem  = n_total % (ANGLE_BINS * SEVERITY_BINS)
    plan = []
    for r in range(ANGLE_BINS):
        for c in range(SEVERITY_BINS):
            plan.extend([(r, c)] * base)
    for k in range(rem):
        plan.append((k // SEVERITY_BINS, k % SEVERITY_BINS))
    return plan


def angle_bin_to_target(angle_bin, angle_edges):
    return float(np.random.uniform(angle_edges[angle_bin], angle_edges[angle_bin + 1]))


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def make_exo_env(sigma=0.0):
    base  = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    inner = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=str(HEALTHY_PATH),
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )
    if sigma > 0.0:
        return NoisyExoWrapper(inner, noise_sigma=sigma, randomize_sigma=False)
    return inner


def make_healthy_env():
    return gym.make("myoElbowPose1D6MRandom-v0")


def load_exo_policy(path, is_recurrent):
    if is_recurrent:
        if not _HAS_RECPPO:
            raise ImportError("sb3_contrib required for RecurrentPPO")
        return RecurrentPPO.load(path)
    return PPO.load(path)


def get_base_env(env):
    if isinstance(env, NoisyExoWrapper):
        return env.env
    return env


# ---------------------------------------------------------------------------
# Healthy reference track
# ---------------------------------------------------------------------------
def run_healthy_track(healthy_env, healthy_policy, target_angle):
    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = [target_angle]
    healthy_env.unwrapped.target_type      = "fixed"
    healthy_env.unwrapped.update_target(restore_sim=True)
    healthy_env.unwrapped.sim.data.qpos[:] = 0.0
    healthy_env.unwrapped.sim.data.qvel[:] = 0.0
    healthy_env.unwrapped.sim.forward()

    obs    = healthy_env.unwrapped.get_obs()[:healthy_policy.observation_space.shape[0]]
    angles = []
    for _ in range(MAX_STEPS):
        action, _ = healthy_policy.predict(obs, deterministic=True)
        next_obs, _, done, truncated, _ = healthy_env.step(action)
        obs = next_obs[:healthy_policy.observation_space.shape[0]]
        angles.append(float(healthy_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles


def _safe_r(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    r, _ = pearsonr(a[:n], b[:n])
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Patient state configuration
# ---------------------------------------------------------------------------
def configure_patient(base_env, target_angle, force_scale, activation_slowdown,
                      mf_vals, split_vals):
    base_env.base_env.unwrapped.target_jnt_value = [target_angle]
    base_env.base_env.unwrapped.target_type = "fixed"
    base_env.base_env.unwrapped.update_target(restore_sim=True)
    remaining = 1.0 - mf_vals
    base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split_vals
    base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split_vals)
    base_env.base_env.unwrapped.muscle_fatigue.MF[:] = mf_vals
    base_env.force_scale         = force_scale
    base_env.activation_slowdown = activation_slowdown
    base_env._apply_brady()
    base_env.base_env.unwrapped.sim.data.qpos[:] = 0.0
    base_env.base_env.unwrapped.sim.data.qvel[:] = 0.0
    base_env.base_env.unwrapped.sim.forward()
    raw = base_env._current_raw_obs()
    return base_env._build_obs(raw)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------
def run_exo_episode(exo_env, base_env, policy, obs, is_recurrent):
    angles        = []
    total_reward  = 0.0
    goal_achieved = False
    goal_time     = float("nan")
    lstm_states   = None
    ep_start      = np.ones((1,), dtype=bool)

    for step in range(MAX_STEPS):
        if is_recurrent:
            action, lstm_states = policy.predict(
                obs, state=lstm_states, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, reward, done, truncated, _ = exo_env.step(action)
        total_reward += float(reward)
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))

        solved = base_env.base_env.unwrapped.rwd_dict.get("solved", False)
        if bool(np.asarray(solved).flat[0]) and not goal_achieved:
            goal_achieved = True
            goal_time     = step * base_env.base_env.unwrapped.dt

        if done or truncated:
            break

    return {"angles": angles, "reward": total_reward,
            "goal_achieved": int(goal_achieved), "goal_time": goal_time,
            "steps": len(angles)}


def run_noexo_episode(exo_env, base_env):
    zero  = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    angles        = []
    total_reward  = 0.0
    goal_achieved = False
    goal_time     = float("nan")

    for step in range(MAX_STEPS):
        _, reward, done, truncated, _ = exo_env.step(zero)
        total_reward += float(reward)
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))

        solved = base_env.base_env.unwrapped.rwd_dict.get("solved", False)
        if bool(np.asarray(solved).flat[0]) and not goal_achieved:
            goal_achieved = True
            goal_time     = step * base_env.base_env.unwrapped.dt

        if done or truncated:
            break

    return {"angles": angles, "reward": total_reward,
            "goal_achieved": int(goal_achieved), "goal_time": goal_time,
            "steps": len(angles)}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_policy_evaluation(policy_name, policy_path, is_recurrent,
                           n_episodes, sigma=0.0, skip_if_exists=False):
    csv_path = OUT_RAW / f"{policy_name}_trials.csv"
    if skip_if_exists and csv_path.exists():
        print(f"  [skip] {policy_name} -- CSV exists, loading...")
        return _load_trials_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"Evaluating: {policy_name}")
    print(f"  episodes={n_episodes}  recurrent={is_recurrent}  sigma={sigma:.3f}")
    print(f"{'='*60}")

    is_healthy_only = (policy_path is None and policy_name == "healthy_policy")
    is_noexo        = (policy_path is None and policy_name == "no_exo")

    healthy_policy = PPO.load(str(HEALTHY_PATH))
    exo_policy     = None
    if not (is_healthy_only or is_noexo):
        exo_policy = load_exo_policy(policy_path, is_recurrent)

    healthy_env = make_healthy_env()
    exo_env     = make_exo_env(sigma)
    base_env    = get_base_env(exo_env)

    tmp         = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    low         = float(tmp.unwrapped.target_jnt_range[0, 0])
    high        = float(tmp.unwrapped.target_jnt_range[0, 1])
    tmp.close()
    angle_edges = np.linspace(low, high, ANGLE_BINS + 1)

    trial_plan = plan_trials(n_episodes)
    n_muscles  = base_env.n_muscles
    trials     = []
    t0_wall    = time.time()

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRIAL_CSV_HEADER)
        writer.writeheader()

        for i, (angle_bin, sev_q) in enumerate(trial_plan):
            target_angle = angle_bin_to_target(angle_bin, angle_edges)
            fs_r, sl_r, mf_r = severity_quartile_to_range(sev_q)
            force_scale    = float(np.random.uniform(*fs_r))
            act_slow       = float(np.random.uniform(*sl_r))
            avg_mf_t       = float(np.random.uniform(*mf_r))
            mf_vals        = np.random.uniform(
                max(avg_mf_t * 0.9, 0.0), min(avg_mf_t * 1.1, 1.0), size=n_muscles)
            split_vals     = np.random.uniform(0.0, 1.0, size=n_muscles)
            actual_avg_mf  = float(np.mean(mf_vals))
            severity       = compute_severity(force_scale, act_slow, actual_avg_mf)

            healthy_angles = run_healthy_track(healthy_env, healthy_policy, target_angle)

            if is_healthy_only:
                # Healthy policy evaluated on healthy env -- run a second time to get reward
                h2 = make_healthy_env()
                h2.reset()
                h2.unwrapped.target_jnt_value = [target_angle]
                h2.unwrapped.target_type = "fixed"
                h2.unwrapped.update_target(restore_sim=True)
                h2.unwrapped.sim.data.qpos[:] = 0.0
                h2.unwrapped.sim.data.qvel[:] = 0.0
                h2.unwrapped.sim.forward()
                h_obs = h2.unwrapped.get_obs()[:healthy_policy.observation_space.shape[0]]
                ep_r  = 0.0
                ep_g  = False
                ep_gt = float("nan")
                ep_s  = 0
                for step in range(MAX_STEPS):
                    action, _ = healthy_policy.predict(h_obs, deterministic=True)
                    n_obs, rwd, done, trunc, _ = h2.step(action)
                    h_obs  = n_obs[:healthy_policy.observation_space.shape[0]]
                    ep_r  += float(rwd)
                    ep_s  += 1
                    solved = h2.unwrapped.rwd_dict.get("solved", False)
                    if bool(np.asarray(solved).flat[0]) and not ep_g:
                        ep_g  = True
                        ep_gt = step * h2.unwrapped.dt
                    if done or trunc:
                        break
                h2.close()
                result = {"reward": ep_r, "goal_achieved": int(ep_g),
                          "goal_time": ep_gt, "steps": ep_s, "angles": healthy_angles}
                pearson_r = 1.0   # reference vs itself
            else:
                exo_env.reset()
                obs = configure_patient(base_env, target_angle, force_scale,
                                        act_slow, mf_vals, split_vals)
                if is_noexo:
                    result = run_noexo_episode(exo_env, base_env)
                else:
                    result = run_exo_episode(exo_env, base_env, exo_policy,
                                             obs, is_recurrent)
                pearson_r = _safe_r(result["angles"], healthy_angles)

            row = {
                "trial_idx":          i,
                "angle_bin":          angle_bin,
                "severity_bin":       sev_q,
                "target_angle":       f"{target_angle:.6f}",
                "force_scale":        f"{force_scale:.6f}",
                "activation_slowdown": f"{act_slow:.6f}",
                "avg_mf":             f"{actual_avg_mf:.6f}",
                "severity":           f"{severity:.6f}",
                "pearson_r":          f"{pearson_r:.6f}",
                "reward":             f"{result['reward']:.6f}",
                "goal_achieved":      result["goal_achieved"],
                "goal_time_s":        f"{result['goal_time']:.6f}"
                                      if not np.isnan(float(result["goal_time"])) else "",
                "episode_steps":      result["steps"],
            }
            trials.append(row)
            writer.writerow(row)
            f.flush()

            if (i + 1) % 200 == 0 or (i + 1) == n_episodes:
                elapsed = time.time() - t0_wall
                eta     = elapsed / (i + 1) * (n_episodes - i - 1)
                print(f"  [{i+1:4d}/{n_episodes}]  "
                      f"r={pearson_r:.3f}  "
                      f"goal={result['goal_achieved']}  "
                      f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    healthy_env.close()
    exo_env.close()
    print(f"  Saved -> {csv_path}")
    return trials


def _load_trials_csv(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def pct_stats(values):
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {k: float("nan")
                for k in ["mean","std","min","max","p5","p25","p50","p75","p95","p99","n"]}
    return {
        "mean": float(np.mean(values)),
        "std":  float(np.std(values)),
        "min":  float(np.min(values)),
        "max":  float(np.max(values)),
        "p5":   float(np.percentile(values, 5)),
        "p25":  float(np.percentile(values, 25)),
        "p50":  float(np.percentile(values, 50)),
        "p75":  float(np.percentile(values, 75)),
        "p95":  float(np.percentile(values, 95)),
        "p99":  float(np.percentile(values, 99)),
        "n":    len(values),
    }


def trials_to_arrays(trials):
    def _f(key):
        out = []
        for t in trials:
            v = t.get(key, "")
            try:
                out.append(float(v))
            except (ValueError, TypeError):
                out.append(float("nan"))
        return np.array(out)

    return {k: _f(k) for k in [
        "pearson_r", "reward", "goal_achieved", "goal_time_s",
        "episode_steps", "severity", "angle_bin", "severity_bin"
    ]}


def compute_full_stats(policy_name, trials):
    arrs  = trials_to_arrays(trials)
    stats = {"policy": policy_name, "n_episodes": len(trials)}

    for metric in ["pearson_r", "reward", "goal_achieved", "episode_steps"]:
        s = pct_stats(arrs[metric])
        for k, v in s.items():
            stats[f"{metric}_{k}"] = round(v, 6) if isinstance(v, float) else v

    for q in range(SEVERITY_BINS):
        mask = arrs["severity_bin"] == q
        pr   = arrs["pearson_r"][mask]; pr = pr[~np.isnan(pr)]
        g    = arrs["goal_achieved"][mask]
        r    = arrs["reward"][mask]; r = r[~np.isnan(r)]
        stats[f"pearson_r_Q{q+1}_mean"] = round(float(np.mean(pr)), 6) if len(pr) else float("nan")
        stats[f"pearson_r_Q{q+1}_std"]  = round(float(np.std(pr)),  6) if len(pr) else float("nan")
        stats[f"pearson_r_Q{q+1}_n"]    = len(pr)
        stats[f"goal_rate_Q{q+1}"]      = round(float(np.mean(g)),  6) if len(g)  else float("nan")
        stats[f"reward_Q{q+1}_mean"]    = round(float(np.mean(r)),  6) if len(r)  else float("nan")

    for b in range(ANGLE_BINS):
        mask = arrs["angle_bin"] == b
        pr   = arrs["pearson_r"][mask]; pr = pr[~np.isnan(pr)]
        g    = arrs["goal_achieved"][mask]
        r    = arrs["reward"][mask]; r = r[~np.isnan(r)]
        stats[f"pearson_r_A{b+1}_mean"] = round(float(np.mean(pr)), 6) if len(pr) else float("nan")
        stats[f"pearson_r_A{b+1}_std"]  = round(float(np.std(pr)),  6) if len(pr) else float("nan")
        stats[f"pearson_r_A{b+1}_n"]    = len(pr)
        stats[f"goal_rate_A{b+1}"]      = round(float(np.mean(g)),  6) if len(g)  else float("nan")
        stats[f"reward_A{b+1}_mean"]    = round(float(np.mean(r)),  6) if len(r)  else float("nan")

    for ab in range(ANGLE_BINS):
        for sq in range(SEVERITY_BINS):
            mask = (arrs["angle_bin"] == ab) & (arrs["severity_bin"] == sq)
            pr   = arrs["pearson_r"][mask]; pr = pr[~np.isnan(pr)]
            g    = arrs["goal_achieved"][mask]
            r    = arrs["reward"][mask]; r = r[~np.isnan(r)]
            key  = f"heatmap_A{ab+1}_Q{sq+1}"
            stats[key + "_pearson_r"] = round(float(np.mean(pr)), 6) if len(pr) else float("nan")
            stats[key + "_goal_rate"] = round(float(np.mean(g)),  6) if len(g)  else float("nan")
            stats[key + "_reward"]    = round(float(np.mean(r)),  6) if len(r)  else float("nan")
            stats[key + "_n"]         = len(pr)

    return stats


# ---------------------------------------------------------------------------
# Per-policy CSV helpers
# ---------------------------------------------------------------------------
def save_per_cell_csv(policy_name, trials):
    arrs = trials_to_arrays(trials)
    rows = []
    for ab in range(ANGLE_BINS):
        for sq in range(SEVERITY_BINS):
            mask = (arrs["angle_bin"] == ab) & (arrs["severity_bin"] == sq)
            pr   = arrs["pearson_r"][mask]; pr = pr[~np.isnan(pr)]
            g    = arrs["goal_achieved"][mask]
            r    = arrs["reward"][mask]; r = r[~np.isnan(r)]
            s_pr = pct_stats(pr)
            rows.append({
                "angle_bin":      ANGLE_LABELS[ab],
                "severity_bin":   SEV_LABELS[sq],
                "n":              len(pr),
                "pearson_r_mean": round(s_pr["mean"], 6),
                "pearson_r_std":  round(s_pr["std"],  6),
                "pearson_r_p5":   round(s_pr["p5"],   6),
                "pearson_r_p50":  round(s_pr["p50"],  6),
                "pearson_r_p95":  round(s_pr["p95"],  6),
                "goal_rate":      round(float(np.mean(g)), 6) if len(g) else float("nan"),
                "reward_mean":    round(float(np.mean(r)), 6) if len(r) else float("nan"),
                "reward_std":     round(float(np.std(r)),  6) if len(r) else float("nan"),
            })
    path = OUT_SUMMARIES / f"{policy_name}_per_cell.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def save_per_quartile_csv(policy_name, trials):
    arrs = trials_to_arrays(trials)
    rows = []
    for q in range(SEVERITY_BINS):
        mask = arrs["severity_bin"] == q
        pr   = arrs["pearson_r"][mask]; pr = pr[~np.isnan(pr)]
        s    = pct_stats(pr)
        g    = arrs["goal_achieved"][mask]
        r    = arrs["reward"][mask]; r = r[~np.isnan(r)]
        row  = {"severity_quartile": SEV_LABELS[q]}
        row.update({f"pearson_r_{k}": round(v, 6) if isinstance(v, float) else v
                    for k, v in s.items()})
        row["goal_rate"]   = round(float(np.mean(g)), 6) if len(g) else float("nan")
        row["reward_mean"] = round(float(np.mean(r)), 6) if len(r) else float("nan")
        row["reward_std"]  = round(float(np.std(r)),  6) if len(r) else float("nan")
        rows.append(row)
    path = OUT_SUMMARIES / f"{policy_name}_per_quartile.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def save_per_angle_csv(policy_name, trials):
    arrs = trials_to_arrays(trials)
    rows = []
    for ab in range(ANGLE_BINS):
        mask = arrs["angle_bin"] == ab
        pr   = arrs["pearson_r"][mask]; pr = pr[~np.isnan(pr)]
        s    = pct_stats(pr)
        g    = arrs["goal_achieved"][mask]
        r    = arrs["reward"][mask]; r = r[~np.isnan(r)]
        row  = {"angle_bin": ANGLE_LABELS[ab]}
        row.update({f"pearson_r_{k}": round(v, 6) if isinstance(v, float) else v
                    for k, v in s.items()})
        row["goal_rate"]   = round(float(np.mean(g)), 6) if len(g) else float("nan")
        row["reward_mean"] = round(float(np.mean(r)), 6) if len(r) else float("nan")
        row["reward_std"]  = round(float(np.std(r)),  6) if len(r) else float("nan")
        rows.append(row)
    path = OUT_SUMMARIES / f"{policy_name}_per_angle.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def plot_heatmap(policy_name, trials, metric="pearson_r",
                 vmin=0.0, vmax=1.0, cbar_label="Mean Pearson r vs Healthy"):
    arrs   = trials_to_arrays(trials)
    matrix = np.full((ANGLE_BINS, SEVERITY_BINS), np.nan)
    for ab in range(ANGLE_BINS):
        for sq in range(SEVERITY_BINS):
            mask = (arrs["angle_bin"] == ab) & (arrs["severity_bin"] == sq)
            vals = arrs[metric][mask]; vals = vals[~np.isnan(vals)]
            if len(vals):
                matrix[ab, sq] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(SEVERITY_BINS));  ax.set_xticklabels(SEV_LABELS)
    ax.set_yticks(range(ANGLE_BINS));     ax.set_yticklabels(ANGLE_LABELS)
    ax.set_xlabel("Severity Quartile");   ax.set_ylabel("Target Angle (rad)")
    ax.set_title(f"{policy_name}  --  {cbar_label}")
    plt.colorbar(im, ax=ax, label=cbar_label)
    for i in range(ANGLE_BINS):
        for j in range(SEVERITY_BINS):
            v = matrix[i, j]
            txt   = f"{v:.3f}" if not np.isnan(v) else "N/A"
            color = "white" if v > (vmin + (vmax - vmin) * 0.6) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color)
    plt.tight_layout()
    tag  = "reward_heatmap" if metric == "reward" else "heatmap"
    path = OUT_PLOTS / f"{policy_name}_{tag}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight");  plt.close()
    print(f"  Plot -> {path.name}")


def plot_comparison_bar(all_stats, metric, ylabel, title, fname):
    names  = list(all_stats.keys())
    means  = [all_stats[n].get(f"{metric}_mean", float("nan")) for n in names]
    stds   = [all_stats[n].get(f"{metric}_std",  float("nan")) for n in names]
    x      = np.arange(len(names))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(names)))
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.6), 6))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, m in zip(bars, means):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot -> {fname}")


def plot_quartile_line(policy_results, fname, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(policy_results)))
    for color, (pname, stats) in zip(colors, policy_results.items()):
        means = [stats.get(f"pearson_r_Q{q+1}_mean", float("nan")) for q in range(4)]
        stds  = [stats.get(f"pearson_r_Q{q+1}_std",  0.0)           for q in range(4)]
        ax.plot(range(4), means, "o-", color=color, linewidth=2, label=pname)
        ax.fill_between(range(4),
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.12)
    ax.set_xticks(range(4)); ax.set_xticklabels(SEV_LABELS)
    ax.set_ylabel("Mean Pearson r"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot -> {fname}")


def plot_angle_bin_line(policy_results, fname, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(policy_results)))
    for color, (pname, stats) in zip(colors, policy_results.items()):
        means = [stats.get(f"pearson_r_A{b+1}_mean", float("nan")) for b in range(4)]
        stds  = [stats.get(f"pearson_r_A{b+1}_std",  0.0)           for b in range(4)]
        ax.plot(range(4), means, "s-", color=color, linewidth=2, label=pname)
        ax.fill_between(range(4),
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color=color, alpha=0.12)
    ax.set_xticks(range(4)); ax.set_xticklabels(ANGLE_LABELS)
    ax.set_ylabel("Mean Pearson r"); ax.set_title(title)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot -> {fname}")


def plot_ablation_panel(nameA, statsA, nameB, statsB, ablation_title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(ablation_title, fontsize=12, fontweight="bold")

    # Panel 1: bar chart
    ax = axes[0]
    pr_means = [statsA["pearson_r_mean"], statsB["pearson_r_mean"]]
    pr_stds  = [statsA["pearson_r_std"],  statsB["pearson_r_std"]]
    gr_means = [statsA["goal_achieved_mean"], statsB["goal_achieved_mean"]]
    rw_means = [statsA["reward_mean"], statsB["reward_mean"]]
    x = np.arange(2); w = 0.25
    colors = ["steelblue", "coral"]
    b1 = ax.bar(x - w, pr_means, w, yerr=pr_stds, capsize=4,
                color=colors, alpha=0.85, label="Pearson r")
    ax2 = ax.twinx()
    ax2.bar(x,     gr_means, w, color=colors, alpha=0.45, hatch="//", label="Goal rate")
    ax.set_xticks(x); ax.set_xticklabels([nameA, nameB], rotation=15, ha="right")
    ax.set_ylabel("Mean Pearson r"); ax2.set_ylabel("Goal rate")
    ax.set_ylim(0, 1.15);            ax2.set_ylim(0, 1.15)
    ax.set_title("Overall Metrics")
    for bar, v in zip(b1, pr_means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)

    # Panel 2: quartile breakdown
    ax = axes[1]
    for name, stats, color in [(nameA, statsA, "steelblue"), (nameB, statsB, "coral")]:
        m = [stats.get(f"pearson_r_Q{q+1}_mean", float("nan")) for q in range(4)]
        s = [stats.get(f"pearson_r_Q{q+1}_std",  0.0)           for q in range(4)]
        ax.plot(range(4), m, "o-", color=color, linewidth=2, label=name)
        ax.fill_between(range(4),
                        [a-b for a,b in zip(m,s)],
                        [a+b for a,b in zip(m,s)],
                        color=color, alpha=0.12)
    ax.set_xticks(range(4)); ax.set_xticklabels(SEV_LABELS, rotation=10)
    ax.set_ylabel("Pearson r"); ax.set_title("Pearson r by Severity Quartile")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 3: angle-bin breakdown
    ax = axes[2]
    for name, stats, color in [(nameA, statsA, "steelblue"), (nameB, statsB, "coral")]:
        m = [stats.get(f"pearson_r_A{b+1}_mean", float("nan")) for b in range(4)]
        s = [stats.get(f"pearson_r_A{b+1}_std",  0.0)           for b in range(4)]
        ax.plot(range(4), m, "s-", color=color, linewidth=2, label=name)
        ax.fill_between(range(4),
                        [a-b for a,b in zip(m,s)],
                        [a+b for a,b in zip(m,s)],
                        color=color, alpha=0.12)
    ax.set_xticks(range(4)); ax.set_xticklabels(ANGLE_LABELS)
    ax.set_ylabel("Pearson r"); ax.set_title("Pearson r by Target Angle")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOTS / fname, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Plot -> {fname}")


# ---------------------------------------------------------------------------
# Noise ablation (Step 3 Ablation 3)
# ---------------------------------------------------------------------------
def run_noise_ablation(n_episodes_per_sigma, skip_if_exists=False):
    csv_path = OUT_SUMMARIES / "noise_ablation.csv"
    if skip_if_exists and csv_path.exists():
        print("[skip] noise_ablation.csv exists")
        return

    sigmas   = [0.00, 0.01, 0.05, 0.10]
    policies = [
        ("RecPPO_clean",        str(POLICIES_DIR / "policy_brady_deg_recurrent.zip"),       True),
        ("RecPPO_noisy_trained", str(POLICIES_DIR / "policy_brady_deg_recurrent_noisy.zip"), True),
    ]

    results = {}
    for pname, ppath, is_rec in policies:
        if not os.path.exists(ppath):
            print(f"  [skip] {pname} -- not found"); continue
        print(f"\n  Noise ablation: {pname}")
        policy         = load_exo_policy(ppath, is_rec)
        healthy_policy = PPO.load(str(HEALTHY_PATH))
        healthy_env    = make_healthy_env()
        tmp  = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
        low  = float(tmp.unwrapped.target_jnt_range[0, 0])
        high = float(tmp.unwrapped.target_jnt_range[0, 1])
        tmp.close()
        angle_edges = np.linspace(low, high, ANGLE_BINS + 1)
        trial_plan  = plan_trials(n_episodes_per_sigma)
        results[pname] = {}

        for sigma in sigmas:
            exo_env  = make_exo_env(sigma)
            base_env = get_base_env(exo_env)
            n_mus    = base_env.n_muscles
            pr_list  = []; g_list = []; r_list = []

            for angle_bin, sev_q in trial_plan:
                ta = angle_bin_to_target(angle_bin, angle_edges)
                fs_r, sl_r, mf_r = severity_quartile_to_range(sev_q)
                fs  = float(np.random.uniform(*fs_r))
                sl  = float(np.random.uniform(*sl_r))
                mft = float(np.random.uniform(*mf_r))
                mfv = np.random.uniform(max(mft*0.9,0.0), min(mft*1.1,1.0), size=n_mus)
                spv = np.random.uniform(0.0, 1.0, size=n_mus)
                h_a = run_healthy_track(healthy_env, healthy_policy, ta)
                exo_env.reset()
                obs = configure_patient(base_env, ta, fs, sl, mfv, spv)
                res = run_exo_episode(exo_env, base_env, policy, obs, is_rec)
                pr_list.append(_safe_r(res["angles"], h_a))
                g_list.append(res["goal_achieved"])
                r_list.append(res["reward"])
            exo_env.close()

            results[pname][sigma] = {
                "pearson_r_mean": round(float(np.mean(pr_list)), 6),
                "pearson_r_std":  round(float(np.std(pr_list)),  6),
                "goal_rate":      round(float(np.mean(g_list)),  6),
                "reward_mean":    round(float(np.mean(r_list)),  6),
                "n":              len(pr_list),
            }
            r_val = results[pname][sigma]
            print(f"    sigma={sigma:.2f}  r={r_val['pearson_r_mean']:.4f}"
                  f"  goal={r_val['goal_rate']:.3f}")
        healthy_env.close()

    # Save
    rows = [{"policy": pn, "sigma": sig, **m}
            for pn, sig_d in results.items()
            for sig, m in sig_d.items()]
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)

    # Plot
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Noise Ablation: RecPPO Clean vs Noise-Trained", fontsize=12)
        colors = {"RecPPO_clean": "steelblue", "RecPPO_noisy_trained": "crimson"}
        for ax_i, (met, ylabel, title) in enumerate([
            ("pearson_r_mean", "Mean Pearson r vs Healthy", "Pearson r at sigma"),
            ("goal_rate",      "Goal Achievement Rate",     "Goal Rate at sigma"),
        ]):
            ax = axes[ax_i]
            for pname, sig_d in results.items():
                xs = sorted(sig_d.keys())
                ys = [sig_d[s][met] for s in xs]
                stds = [sig_d[s].get("pearson_r_std", 0.0) for s in xs] \
                       if met == "pearson_r_mean" else None
                color = colors.get(pname, "gray")
                ax.plot(xs, ys, "o-", linewidth=2, color=color, label=pname)
                if stds:
                    ax.fill_between(xs, [y-s for y,s in zip(ys,stds)],
                                    [y+s for y,s in zip(ys,stds)],
                                    color=color, alpha=0.15)
                for x, y in zip(xs, ys):
                    ax.annotate(f"{y:.3f}", (x, y),
                                textcoords="offset points", xytext=(0, 8),
                                ha="center", fontsize=8)
            ax.set_xlabel("EMG Noise sigma"); ax.set_ylabel(ylabel)
            ax.set_title(title); ax.legend(fontsize=9)
            ax.set_xticks(sigmas); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / "noise_ablation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Plot -> noise_ablation.png")

    return results


# ---------------------------------------------------------------------------
# Training curve stats (Step 4)
# ---------------------------------------------------------------------------
def compute_training_curve_stats():
    log_map = {
        "policy_brady_deg_recurrent":
            ["policy_brady_deg_recurrent_rewards.csv"],
        "policy_deg_recurrent":
            ["policy_deg_recurrent_rewards.csv"],
        "policy_brady_deg":
            ["policy_brady_deg_rewards.csv"],
        "policy_deg":
            ["policy_deg_rewards.csv"],
        "policy_brady_deg_recurrent_noisy":
            ["policy_brady_deg_recurrent_noisy_rewards.csv",
             "policy_brady_deg_recurrent_noisy_finetune_rewards.csv"],
        "policy_brady_deg_lstm":
            ["policy_brady_deg_lstm_rewards.csv"],
        "policy_deg_lstm":
            ["policy_deg_lstm_rewards.csv"],
        "policy_brady_deg_lstm_extraobs":
            ["policy_brady_deg_lstm_extraobs_rewards.csv"],
    }

    rows   = []
    fig, ax = plt.subplots(figsize=(14, 7))
    colors  = plt.cm.tab10(np.linspace(0, 0.9, len(log_map)))

    for color, (pname, files) in zip(colors, log_map.items()):
        all_ts, all_r = [], []
        ts_offset = 0
        for fname in files:
            fpath = LOGS_DIR / fname
            if not fpath.exists():
                print(f"  [warn] Log not found: {fpath}")
                continue
            with open(fpath, newline="") as f:
                for row in csv.DictReader(f):
                    all_ts.append(int(row["timestep"]) + ts_offset)
                    all_r.append(float(row["mean_reward"]))
            if all_ts:
                ts_offset = all_ts[-1]

        if not all_ts:
            continue

        ts_arr   = np.array(all_ts)
        r_arr    = np.array(all_r)
        final_r  = float(r_arr[-1])
        max_r    = float(r_arr.max())
        min_r    = float(r_arr.min())
        total_ts = int(ts_arr[-1])

        thr          = final_r * 0.5
        conv_idx     = np.where(r_arr >= thr)[0]
        conv_ts      = int(ts_arr[conv_idx[0]]) if len(conv_idx) else -1

        n             = len(r_arr)
        last_slice    = r_arr[max(0, int(n * 0.9)):]
        plateau       = bool(np.std(last_slice) < 0.05 * (abs(np.mean(last_slice)) + 1e-9))

        rows.append({
            "policy":             pname,
            "total_timesteps":    total_ts,
            "n_log_entries":      len(ts_arr),
            "final_reward":       round(final_r, 4),
            "max_reward":         round(max_r,   4),
            "min_reward":         round(min_r,   4),
            "mean_reward_all":    round(float(r_arr.mean()), 4),
            "std_reward_all":     round(float(r_arr.std()),  4),
            "50pct_conv_ts":      conv_ts,
            "plateau_last10pct":  plateau,
        })

        # Smoothed curve for plot
        window = max(1, len(r_arr) // 80)
        smooth = np.convolve(r_arr, np.ones(window) / window, mode="valid")
        ts_sm  = ts_arr[window // 2: window // 2 + len(smooth)]
        ax.plot(ts_sm, smooth, linewidth=1.5, color=color,
                label=f"{pname} (final={final_r:.0f})")

    ax.set_xlabel("Training Timestep")
    ax.set_ylabel("Mean Episode Reward (smoothed)")
    ax.set_title("Training Reward Curves -- All Policies")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_CURVES / "reward_curves_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot -> training_curves/reward_curves_overlay.png")

    path = OUT_CURVES / "training_curve_stats.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  Saved -> training_curves/training_curve_stats.csv")
    return rows


# ---------------------------------------------------------------------------
# Pretty print summary
# ---------------------------------------------------------------------------
def print_policy_summary(policy_name, stats):
    print(f"\n{'--'*30}")
    print(f"  Policy    : {policy_name}")
    print(f"  N episodes: {stats['n_episodes']}")
    hdr = f"  {'Metric':20s}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  "
    hdr += f"{'p5':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}"
    print(hdr)
    for metric, label in [
        ("pearson_r",     "Pearson r"),
        ("goal_achieved", "Goal rate"),
        ("reward",        "Reward"),
        ("episode_steps", "Ep. steps"),
    ]:
        vals = [stats.get(f"{metric}_{k}", float("nan"))
                for k in ["mean","std","min","max","p5","p50","p95","p99"]]
        fmt  = f"  {label:20s}" + "".join(f"  {v:8.4f}" for v in vals)
        print(fmt)

    print(f"  Per-severity Pearson r:")
    for q in range(4):
        m = stats.get(f"pearson_r_Q{q+1}_mean", float("nan"))
        s = stats.get(f"pearson_r_Q{q+1}_std",  float("nan"))
        g = stats.get(f"goal_rate_Q{q+1}",       float("nan"))
        r = stats.get(f"reward_Q{q+1}_mean",     float("nan"))
        print(f"    {SEV_LABELS[q]:12s}: r={m:.4f}+/-{s:.4f}  goal={g:.3f}  reward={r:.2f}")
    print(f"  Per-angle Pearson r:")
    for b in range(4):
        m  = stats.get(f"pearson_r_A{b+1}_mean", float("nan"))
        s  = stats.get(f"pearson_r_A{b+1}_std",  float("nan"))
        g  = stats.get(f"goal_rate_A{b+1}",       float("nan"))
        r  = stats.get(f"reward_A{b+1}_mean",     float("nan"))
        print(f"    {ANGLE_LABELS[b]:12s}: r={m:.4f}+/-{s:.4f}  goal={g:.3f}  reward={r:.2f}")
    print(f"  Heatmap (angle x severity, Pearson r):")
    header_row = f"    {'':12s}" + "".join(f"  {lb:12s}" for lb in SEV_LABELS)
    print(header_row)
    for ab in range(ANGLE_BINS):
        row_str = f"    {ANGLE_LABELS[ab]:12s}"
        for sq in range(SEVERITY_BINS):
            v = stats.get(f"heatmap_A{ab+1}_Q{sq+1}_pearson_r", float("nan"))
            row_str += f"  {v:12.4f}"
        print(row_str)


# ---------------------------------------------------------------------------
# Policy registry
# ---------------------------------------------------------------------------
POLICY_REGISTRY = [
    # (name, path_or_None, is_recurrent, sigma)
    ("policy_brady_deg_recurrent",
     "policy_brady_deg_recurrent.zip",       True,  0.0),
    ("policy_deg_recurrent",
     "policy_deg_recurrent.zip",             True,  0.0),
    ("policy_brady_deg",
     "policy_brady_deg.zip",                 False, 0.0),
    ("policy_deg",
     "policy_deg.zip",                       False, 0.0),
    ("policy_brady_deg_recurrent_noisy",
     "policy_brady_deg_recurrent_noisy.zip", True,  0.0),
    ("no_exo",      None,                    False, 0.0),
    ("healthy_policy", None,                 False, 0.0),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ReGainX full evaluation suite")
    parser.add_argument("--episodes",      type=int, default=4000,
                        help="Total episodes per policy (default: 4000)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip policies whose raw CSV already exists")
    parser.add_argument("--step",          type=int, default=0,
                        help="Run only step N (1=check, 2=eval, 3=ablations, 4=curves). 0=all")
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    N = args.episodes
    cells = ANGLE_BINS * SEVERITY_BINS

    print(f"\n{'='*60}")
    print(f"ReGainX Full Evaluation Suite")
    print(f"  Episodes per policy   : {N}")
    print(f"  Episodes per cell     : {N // cells}  ({cells} cells)")
    print(f"  Output                : {OUT_DIR}")
    print(f"{'='*60}")

    # ---- Step 4: Training curves (fast, independent) -----------------------
    if args.step in (0, 4):
        print(f"\n{'='*60}\nSTEP 4 -- Training Curve Statistics\n{'='*60}")
        curve_rows = compute_training_curve_stats()
        print(f"\n  {'Policy':45s}  {'TotalTS':>10}  {'FinalR':>8}  "
              f"{'MaxR':>8}  {'Conv50%TS':>10}  {'Plateau':>7}")
        print("  " + "-"*90)
        for row in curve_rows:
            print(f"  {row['policy']:45s}  {row['total_timesteps']:10d}  "
                  f"{row['final_reward']:8.2f}  {row['max_reward']:8.2f}  "
                  f"{row['50pct_conv_ts']:10d}  {str(row['plateau_last10pct']):>7}")

    if args.step == 4:
        print("\nStep 4 done.")
        return

    # ---- Step 2: Evaluations -----------------------------------------------
    all_stats  = {}
    all_trials = {}

    if args.step in (0, 1, 2, 3):
        print(f"\n{'='*60}\nSTEP 2 -- Policy Evaluations ({N} episodes each)\n{'='*60}")

        for pname, pfile, is_rec, sigma in POLICY_REGISTRY:
            if pfile is not None and not (POLICIES_DIR / pfile).exists():
                print(f"\n  [skip] {pname} -- file not found")
                continue
            ppath  = str(POLICIES_DIR / pfile) if pfile else None
            trials = run_policy_evaluation(
                policy_name    = pname,
                policy_path    = ppath,
                is_recurrent   = is_rec,
                n_episodes     = N,
                sigma          = sigma,
                skip_if_exists = args.skip_existing,
            )
            all_trials[pname] = trials
            stats             = compute_full_stats(pname, trials)
            all_stats[pname]  = stats

            save_per_cell_csv(pname, trials)
            save_per_quartile_csv(pname, trials)
            save_per_angle_csv(pname, trials)

            plot_heatmap(pname, trials, metric="pearson_r",
                         vmin=0.0, vmax=1.0, cbar_label="Mean Pearson r vs Healthy")
            try:
                rwd = trials_to_arrays(trials)["reward"]
                if not np.all(np.isnan(rwd)):
                    vmax = float(np.nanpercentile(rwd, 95))
                    vmin = float(np.nanmin(rwd))
                    plot_heatmap(pname, trials, metric="reward",
                                 vmin=vmin, vmax=vmax, cbar_label="Mean Episode Reward")
            except Exception:
                pass

            print_policy_summary(pname, stats)

        # Master summary CSV
        if all_stats:
            fieldnames = sorted({k for s in all_stats.values() for k in s.keys()})
            path = OUT_SUMMARIES / "all_policies_summary.csv"
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                w.writeheader()
                for s in all_stats.values():
                    w.writerow(s)
            print(f"\n  Saved master summary -> {path}")

        # Comparison plots
        if all_stats:
            plot_comparison_bar(all_stats, "pearson_r",
                                "Mean Pearson r", "Trajectory Recovery -- All Policies",
                                "comparison_pearsonr_bar.png")
            plot_comparison_bar(all_stats, "goal_achieved",
                                "Goal Rate",      "Goal Achievement -- All Policies",
                                "comparison_goalrate_bar.png")
            plot_comparison_bar(all_stats, "reward",
                                "Mean Reward",    "Episode Reward -- All Policies",
                                "comparison_reward_bar.png")
            plot_quartile_line(all_stats, "quartile_line_all.png",
                               "Pearson r by Severity -- All Policies")
            plot_angle_bin_line(all_stats, "angle_bin_line_all.png",
                                "Pearson r by Angle Bin -- All Policies")

    # ---- Step 3: Ablations -------------------------------------------------
    if args.step in (0, 3):
        print(f"\n{'='*60}\nSTEP 3 -- Ablation Studies\n{'='*60}")

        # Ablation 1: training on brady+deg vs deg-only (RecPPO)
        if "policy_brady_deg_recurrent" in all_stats and \
                "policy_deg_recurrent" in all_stats:
            A = all_stats["policy_brady_deg_recurrent"]
            B = all_stats["policy_deg_recurrent"]
            plot_ablation_panel("RecPPO brady+deg", A, "RecPPO deg-only", B,
                                "Ablation 1 -- Training Impairment Complexity (RecPPO)",
                                "ablation1_impairment_complexity_recppo.png")
            print("\nAblation 1 -- Pearson r gap per severity quartile:")
            print(f"  {'Quartile':12s}  {'RecPPO brady+deg':>18}  "
                  f"{'RecPPO deg-only':>16}  {'Gap (brady-deg)':>16}")
            for q in range(4):
                ra = A.get(f"pearson_r_Q{q+1}_mean", float("nan"))
                rb = B.get(f"pearson_r_Q{q+1}_mean", float("nan"))
                print(f"  {SEV_LABELS[q]:12s}  {ra:18.4f}  {rb:16.4f}  {ra-rb:+16.4f}")
            print(f"\n  Overall gap: {A['pearson_r_mean'] - B['pearson_r_mean']:+.4f}")
            print(f"  Goal rate gap: {A['goal_achieved_mean'] - B['goal_achieved_mean']:+.4f}")

        # Ablation 2: RecPPO vs MLP
        if "policy_brady_deg_recurrent" in all_stats and \
                "policy_brady_deg" in all_stats:
            A = all_stats["policy_brady_deg_recurrent"]
            B = all_stats["policy_brady_deg"]
            plot_ablation_panel("RecPPO brady+deg", A, "MLP brady+deg", B,
                                "Ablation 2 -- RecurrentPPO vs MLP on Combined PD",
                                "ablation2_recppo_vs_mlp.png")
            print("\nAblation 2 -- RecPPO vs MLP:")
            for metric, label in [("pearson_r", "Pearson r"),
                                   ("goal_achieved", "Goal rate"),
                                   ("reward", "Reward")]:
                va = A.get(f"{metric}_mean", float("nan"))
                vb = B.get(f"{metric}_mean", float("nan"))
                print(f"  {label:12s}: RecPPO={va:.4f}  MLP={vb:.4f}  gap={va-vb:+.4f}")

        # Ablation 3: noise robustness
        print(f"\nAblation 3 -- Noise Robustness")
        noise_res = run_noise_ablation(
            n_episodes_per_sigma = max(N // 4, 100),
            skip_if_exists       = args.skip_existing,
        )

        # Ablation 4: MLP brady+deg vs MLP deg-only
        if "policy_brady_deg" in all_stats and "policy_deg" in all_stats:
            A = all_stats["policy_brady_deg"]
            B = all_stats["policy_deg"]
            plot_ablation_panel("MLP brady+deg", A, "MLP deg-only", B,
                                "Ablation 4 -- Impairment Complexity Effect on MLP",
                                "ablation4_mlp_impairment_complexity.png")
            print("\nAblation 4 -- MLP brady+deg vs MLP deg-only:")
            for metric, label in [("pearson_r", "Pearson r"),
                                   ("goal_achieved", "Goal rate"),
                                   ("reward", "Reward")]:
                va = A.get(f"{metric}_mean", float("nan"))
                vb = B.get(f"{metric}_mean", float("nan"))
                print(f"  {label:12s}: brady+deg={va:.4f}  deg-only={vb:.4f}  "
                      f"gap={va-vb:+.4f}")

    # ---- Final summary table -----------------------------------------------
    if all_stats:
        print(f"\n{'='*60}\nFINAL SUMMARY TABLE\n{'='*60}")
        print(f"  {'Policy':45s}  {'Pearson r':>10}  {'Goal rate':>10}  "
              f"{'Reward':>10}  {'N':>6}")
        print("  " + "-"*90)
        for pname, stats in all_stats.items():
            pr = stats.get("pearson_r_mean",     float("nan"))
            gr = stats.get("goal_achieved_mean", float("nan"))
            rw = stats.get("reward_mean",        float("nan"))
            n  = stats.get("n_episodes",         0)
            print(f"  {pname:45s}  {pr:10.4f}  {gr:10.4f}  {rw:10.2f}  {n:6d}")

    print(f"\n{'='*60}")
    print(f"All done. Results saved to: {OUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
