"""
Baseline vs RecPPO policy comparison on the brady+deg environment.

Evaluates two policies on identical 80 episodes (5 per cell of a 4×4
angle-bin × severity-quartile grid):
  - Baseline : policy_deg.zip (MLP, deg-only trained — brady is OOD for it)
  - RecPPO   : policy_brady_deg_recurrent.zip (trained with brady+deg)

Outputs to results/baseline_comparison/:
  confusion_matrix_baseline.png
  confusion_matrix_recppo.png
  confusion_matrix_no_exo.png
  comparison_metrics.png          (4-panel combined figure)
  pearsonr_by_severity.png        (standalone severity-sorted line plot)

Usage:
    python compare_baseline.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import myosuite  # noqa: F401 — registers envs
from myosuite.utils import gym
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT_PPO = True
except ImportError:
    _HAS_RECURRENT_PPO = False

from envs.elbow_env import CombinedExoOnlyWrapper
from evaluation import plan_trials, severity_quartile_to_range, angle_bin_to_target
from utils import (
    plot_confusion_matrix,
    compute_severity,
    get_angle_bin,
    get_severity_quartile,
)

# ---------------------------------------------------------------------------
# Hardcoded policy paths
# ---------------------------------------------------------------------------
HEALTHY_PATH   = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\healthy_policy.zip"
BASELINE_PATH  = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\policy_deg.zip"
RECURRENT_PATH = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\policy_brady_deg_recurrent.zip"
OUT_DIR        = "results/baseline_comparison"

# ---------------------------------------------------------------------------
# Evaluation constants
# ---------------------------------------------------------------------------
N_EPISODES     = 80          # 5 per cell × 16 cells
MAX_STEPS      = 500
ANGLE_BINS     = 4
SEVERITY_BINS  = 4
ANGLE_LABELS   = ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5"]
SEVERITY_LABELS = ["Q1 mild", "Q2", "Q3", "Q4 severe"]


# ---------------------------------------------------------------------------
# Patient configuration
# ---------------------------------------------------------------------------

def configure_patient(
    base_env,
    target_angle: float,
    force_scale: float,
    activation_slowdown: float,
    mf_vals: np.ndarray,
    split_vals: np.ndarray,
) -> np.ndarray:
    """Reset sim to the given patient state; return flat obs vector."""
    base_env.base_env.unwrapped.target_jnt_value = [target_angle]
    base_env.base_env.unwrapped.target_type = "fixed"
    base_env.base_env.unwrapped.update_target(restore_sim=True)

    remaining = 1.0 - mf_vals
    base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split_vals
    base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split_vals)
    base_env.base_env.unwrapped.muscle_fatigue.MF[:] = mf_vals
    base_env.force_scale = force_scale
    base_env.activation_slowdown = activation_slowdown
    base_env._apply_brady()
    base_env.base_env.unwrapped.sim.data.qpos[:] = 0.0
    base_env.base_env.unwrapped.sim.data.qvel[:] = 0.0
    base_env.base_env.unwrapped.sim.forward()

    raw = base_env._current_raw_obs()
    return base_env._build_obs(raw)


# ---------------------------------------------------------------------------
# Healthy track
# ---------------------------------------------------------------------------

def run_healthy_track(healthy_env, healthy_policy, target_angle: float) -> list:
    """Run healthy policy to target. Returns list of qpos[0] values."""
    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = [target_angle]
    healthy_env.unwrapped.target_type = "fixed"
    healthy_env.unwrapped.update_target(restore_sim=True)
    healthy_env.unwrapped.sim.data.qpos[:] = 0.0
    healthy_env.unwrapped.sim.data.qvel[:] = 0.0
    healthy_env.unwrapped.sim.forward()

    obs = healthy_env.unwrapped.get_obs()[: healthy_policy.observation_space.shape[0]]
    angles = []
    for _ in range(MAX_STEPS):
        action, _ = healthy_policy.predict(obs, deterministic=True)
        next_obs, _, done, truncated, _ = healthy_env.step(action)
        obs = next_obs[: healthy_policy.observation_space.shape[0]]
        angles.append(float(healthy_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles


# ---------------------------------------------------------------------------
# Exo track (one policy)
# ---------------------------------------------------------------------------

def run_exo_track(
    exo_env,
    base_env,
    policy,
    target_angle: float,
    force_scale: float,
    activation_slowdown: float,
    mf_vals: np.ndarray,
    split_vals: np.ndarray,
    is_recurrent: bool = False,
) -> dict:
    """
    Run one policy on the exo env with the given patient state.
    Returns angles, cumulative reward, goal_achieved, goal_time.
    """
    exo_env.reset()
    obs = configure_patient(base_env, target_angle, force_scale,
                            activation_slowdown, mf_vals, split_vals)

    angles = []
    total_reward = 0.0
    goal_achieved = False
    goal_time = None
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)

    for step in range(MAX_STEPS):
        if is_recurrent:
            action, lstm_states = policy.predict(
                obs, state=lstm_states,
                episode_start=episode_start, deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, reward, done, truncated, _ = exo_env.step(action)
        total_reward += reward
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))

        solved_val = base_env.base_env.unwrapped.rwd_dict.get("solved", False)
        if bool(np.asarray(solved_val).flat[0]) and not goal_achieved:
            goal_achieved = True
            goal_time = step * base_env.base_env.unwrapped.dt
        if done or truncated:
            break

    return {
        "angles":        angles,
        "reward":        total_reward,
        "goal_achieved": goal_achieved,
        "goal_time":     goal_time,
    }


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def build_matrix(
    trials: list,
    corr_key: str,
    severity_edges: np.ndarray,
) -> np.ndarray:
    """
    Build 4×4 Pearson-r matrix (angle bins × severity quartiles).
    Cells with no trials are NaN. angle_bin must be pre-stored in each trial dict.
    """
    matrix = np.full((ANGLE_BINS, SEVERITY_BINS), np.nan)
    counts = np.zeros((ANGLE_BINS, SEVERITY_BINS), dtype=int)
    sums   = np.zeros((ANGLE_BINS, SEVERITY_BINS))

    for t in trials:
        row = t["angle_bin"]
        col = get_severity_quartile(t["severity"], severity_edges)
        sums[row, col]   += t[corr_key]
        counts[row, col] += 1

    mask = counts >= 1
    matrix[mask] = sums[mask] / counts[mask]
    return matrix


# ---------------------------------------------------------------------------
# Boost calculation
# ---------------------------------------------------------------------------

def compute_boost_pct(acc: float, floor: float) -> float:
    """Percentage of remaining gap from floor to 1.0 that acc fills."""
    gap = max(1.0 - floor, 1e-9)
    return float(np.clip((acc - floor) / gap * 100.0, 0.0, 100.0))


# ---------------------------------------------------------------------------
# Full trial (4 tracks)
# ---------------------------------------------------------------------------

def run_trial(
    exo_env,
    base_env,
    healthy_env,
    baseline_policy,
    recurrent_policy,
    healthy_policy,
    angle_bin: int,
    sev_quartile: int,
    angle_edges: np.ndarray,
) -> dict:
    """
    Run all four tracks for one episode:
      1. Healthy
      2. No-exo (zero action, same patient state)
      3. Baseline (policy_deg)
      4. RecPPO (policy_brady_deg_recurrent)

    Patient state is sampled from the given (angle_bin, sev_quartile) cell
    and applied identically to tracks 2-4.
    """
    # Sample patient state
    target_angle = angle_bin_to_target(angle_bin, angle_edges)
    fs_range, sl_range, mf_range = severity_quartile_to_range(sev_quartile)
    force_scale         = float(np.random.uniform(*fs_range))
    activation_slowdown = float(np.random.uniform(*sl_range))
    avg_mf_target       = float(np.random.uniform(*mf_range))

    n_muscles  = base_env.n_muscles
    mf_vals    = np.random.uniform(
        max(avg_mf_target * 0.9, 0.0),
        min(avg_mf_target * 1.1, 1.0),
        size=n_muscles,
    )
    split_vals = np.random.uniform(0.0, 1.0, size=n_muscles)

    # Track 1: Healthy
    healthy_angles = run_healthy_track(healthy_env, healthy_policy, target_angle)

    actual_avg_mf = float(np.mean(mf_vals))
    severity = compute_severity(force_scale, activation_slowdown, actual_avg_mf)

    # Track 2: No-exo (zero action, same patient state)
    exo_env.reset()
    configure_patient(base_env, target_angle, force_scale,
                      activation_slowdown, mf_vals, split_vals)
    zero_action = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    no_exo_angles = []
    for _ in range(MAX_STEPS):
        _, _, done, truncated, _ = exo_env.step(zero_action)
        no_exo_angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break

    # Track 3: Baseline (MLP deg-only, not recurrent)
    baseline_result = run_exo_track(
        exo_env, base_env, baseline_policy,
        target_angle, force_scale, activation_slowdown,
        mf_vals, split_vals, is_recurrent=False,
    )

    # Track 4: RecPPO
    recurrent_result = run_exo_track(
        exo_env, base_env, recurrent_policy,
        target_angle, force_scale, activation_slowdown,
        mf_vals, split_vals, is_recurrent=True,
    )

    def _safe_r(a, b):
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        raw_r, _ = pearsonr(a[:n], b[:n])
        return float(raw_r) if not np.isnan(raw_r) else 0.0

    return {
        "angle_bin":            angle_bin,
        "sev_quartile":         sev_quartile,
        "target_angle":         target_angle,
        "force_scale":          force_scale,
        "activation_slowdown":  activation_slowdown,
        "avg_mf":               actual_avg_mf,
        "severity":             severity,
        "baseline_corr":        _safe_r(baseline_result["angles"],  healthy_angles),
        "recurrent_corr":       _safe_r(recurrent_result["angles"], healthy_angles),
        "no_exo_corr":          _safe_r(no_exo_angles,             healthy_angles),
        "baseline_reward":      baseline_result["reward"],
        "recurrent_reward":     recurrent_result["reward"],
        "baseline_goal":        baseline_result["goal_achieved"],
        "recurrent_goal":       recurrent_result["goal_achieved"],
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison_metrics(trials: list, no_exo_mean: float, out_dir: str) -> None:
    """
    4-panel combined figure:
      Panel 1 (top-left)   : Reward per episode
      Panel 2 (top-right)  : Mean Pearson r bar chart (baseline vs RecPPO vs no-exo)
      Panel 3 (bottom-left): Pearson r per episode — sequential with gap shading
      Panel 4 (bottom-right): Pearson r per episode — sorted by severity
    """
    n = len(trials)
    eps = np.arange(1, n + 1)

    baseline_rewards  = [t["baseline_reward"]  for t in trials]
    recurrent_rewards = [t["recurrent_reward"] for t in trials]
    baseline_r        = np.array([t["baseline_corr"]    for t in trials])
    recurrent_r       = np.array([t["recurrent_corr"]   for t in trials])
    severities        = [t["severity"]          for t in trials]

    mean_base  = float(np.mean(baseline_r))
    mean_rec   = float(np.mean(recurrent_r))
    boost_base = compute_boost_pct(mean_base, no_exo_mean)
    boost_rec  = compute_boost_pct(mean_rec,  no_exo_mean)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Baseline (MLP deg-only) vs RecPPO on Brady+Deg Environment",
                 fontsize=13, fontweight="bold")

    # --- Panel 1: Reward per episode ---
    ax = axes[0, 0]
    ax.plot(eps, baseline_rewards,  color="steelblue", linewidth=1.2,
            label=f"MLP deg-only  (mean={np.mean(baseline_rewards):.1f})")
    ax.plot(eps, recurrent_rewards, color="coral",     linewidth=1.2,
            label=f"RecPPO brady+deg (mean={np.mean(recurrent_rewards):.1f})")
    ax.axhline(np.mean(baseline_rewards),  color="steelblue", linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.axhline(np.mean(recurrent_rewards), color="coral",     linestyle="--",
               linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward per Episode")
    ax.legend(fontsize=8)

    # --- Panel 2: Mean Pearson r bar chart ---
    ax = axes[0, 1]
    bar_labels = ["MLP deg-only\n(baseline)", "RecPPO\nbrady+deg", "No-Exo\nfloor"]
    values = [mean_base, mean_rec, no_exo_mean]
    colors = ["steelblue", "coral", "gray"]
    bars = ax.bar(bar_labels, values, color=colors, alpha=0.85, width=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Healthy (1.0)")
    for bar, val, boost in zip(bars[:2], [mean_base, mean_rec], [boost_base, boost_rec]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                min(val + 0.02, 1.12),
                f"+{boost:.0f}%\ntoward healthy",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Mean Pearson r (vs Healthy)")
    ax.set_title("Mean Accuracy — Pearson r")
    ax.legend(fontsize=8)

    # --- Panel 3: Pearson r per episode — sequential ---
    ax = axes[1, 0]
    ax.plot(eps, baseline_r,  color="steelblue", linewidth=1.2, label="MLP deg-only")
    ax.plot(eps, recurrent_r, color="coral",     linewidth=1.2, label="RecPPO")
    ax.fill_between(eps, baseline_r, recurrent_r,
                    where=(recurrent_r >= baseline_r), alpha=0.15, color="coral",
                    label="RecPPO advantage")
    ax.fill_between(eps, baseline_r, recurrent_r,
                    where=(recurrent_r < baseline_r), alpha=0.15, color="steelblue",
                    label="Baseline advantage")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Healthy")
    ax.axhline(no_exo_mean, color="gray", linestyle=":", linewidth=0.8,
               label=f"No-exo floor ({no_exo_mean:.3f})")
    ax.set_xlabel("Episode (sequential)")
    ax.set_ylabel("Pearson r (vs Healthy)")
    ax.set_title("Pearson r Per Episode — Sequential")
    ax.legend(fontsize=7)

    # --- Panel 4: Pearson r sorted by severity ---
    sort_idx   = np.argsort(severities)
    sorted_sev = np.array(severities)[sort_idx]
    sorted_br  = baseline_r[sort_idx]
    sorted_rr  = recurrent_r[sort_idx]
    ax = axes[1, 1]
    ax.plot(range(n), sorted_br, color="steelblue", linewidth=1.2, label="MLP deg-only")
    ax.plot(range(n), sorted_rr, color="coral",     linewidth=1.2, label="RecPPO")
    ax.fill_between(range(n), sorted_br, sorted_rr,
                    where=(sorted_rr >= sorted_br), alpha=0.15, color="coral")
    ax.fill_between(range(n), sorted_br, sorted_rr,
                    where=(sorted_rr < sorted_br), alpha=0.15, color="steelblue")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(no_exo_mean, color="gray", linestyle=":", linewidth=0.8)
    q_edges = np.percentile(sorted_sev, [25, 50, 75])
    for q in q_edges:
        idx = np.searchsorted(sorted_sev, q)
        ax.axvline(idx, color="dimgray", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Episodes (sorted mild → severe)")
    ax.set_ylabel("Pearson r (vs Healthy)")
    ax.set_title("Pearson r Sorted by Severity")
    ax.legend(fontsize=7)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "comparison_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")
