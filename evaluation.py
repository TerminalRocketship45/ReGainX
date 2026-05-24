"""
Detailed single-policy evaluation: healthy vs impaired+exo movement similarity.

Runs N trials across 3 parallel tracks per trial:
  1. Healthy patient (myoElbowPose1D6MRandom-v0)
  2. Impaired, no exo (exo torque = 0)
  3. Impaired + exo (exo policy active)

Pearson correlation between healthy and impaired+exo trajectories is the
primary accuracy metric. Trials are evenly distributed across the
(angle_bin x severity_quartile) confusion matrix.

Usage:
    python evaluation.py                                 # 32 trials, default paths
    python evaluation.py --trials 64
    python evaluation.py --extraobs --out-dir results/extraobs_eval
    python evaluation.py --out-dir results/cnn_eval
"""

import argparse
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT_PPO = True
except ImportError:
    _HAS_RECURRENT_PPO = False

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.temporal_buffer import TemporalStackWrapper
from utils import (
    compute_severity, get_angle_bin, get_severity_quartile,
    plot_confusion_matrix, add_text_to_frame, save_video,
)

MAX_STEPS = 500
ANGLE_BINS = 4
SEVERITY_BINS = 4
ANGLE_LABELS = ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5"]
SEVERITY_LABELS = ["Q1 mild", "Q2", "Q3", "Q4 severe"]


# -- Trial planning --

def plan_trials(n_trials: int) -> list:
    """
    Return list of (angle_bin, severity_quartile) assignments, evenly distributed.
    n_trials spread across 16 cells; remainder fills cells with fewest runs.
    """
    cells = ANGLE_BINS * SEVERITY_BINS
    base = n_trials // cells
    remainder = n_trials % cells
    plan = []
    for row in range(ANGLE_BINS):
        for col in range(SEVERITY_BINS):
            plan.extend([(row, col)] * base)
    for k in range(remainder):
        row, col = k // SEVERITY_BINS, k % SEVERITY_BINS
        plan.append((row, col))
    return plan


def severity_quartile_to_range(quartile: int) -> tuple:
    """Map quartile index (0-3) to (force_scale_range, slowdown_range, mf_range)."""
    fs_edges = np.linspace(0.6, 0.9, SEVERITY_BINS + 1)[::-1]  # severe=low force
    sl_edges = np.linspace(1.1, 1.4, SEVERITY_BINS + 1)
    mf_edges = np.linspace(0.0, 1.0, SEVERITY_BINS + 1)

    # Q0=mild (low impairment), Q3=severe
    fs_range = (float(fs_edges[quartile + 1]), float(fs_edges[quartile]))
    sl_range = (float(sl_edges[quartile]), float(sl_edges[quartile + 1]))
    mf_range = (float(mf_edges[quartile]), float(mf_edges[quartile + 1]))
    return fs_range, sl_range, mf_range


def angle_bin_to_target(angle_bin: int, angle_edges: np.ndarray) -> float:
    """Sample a target angle uniformly within the given angle bin."""
    return float(np.random.uniform(angle_edges[angle_bin], angle_edges[angle_bin + 1]))


# -- Single trial --

def run_eval_trial(
    exo_env,
    exo_policy,
    healthy_env,
    healthy_policy: PPO,
    target_angle: float,
    force_scale: float,
    activation_slowdown: float,
    avg_mf_target: float,
    is_recurrent_exo: bool = False,
) -> dict:
    """Run healthy, impaired-no-exo, and impaired+exo tracks for one trial."""
    base_env = exo_env.env if isinstance(exo_env, TemporalStackWrapper) else exo_env
    n_muscles = base_env.n_muscles

    def configure_exo_env() -> float:
        base_env.base_env.unwrapped.target_jnt_value = [target_angle]
        base_env.base_env.unwrapped.target_type = "fixed"
        base_env.base_env.unwrapped.update_target(restore_sim=True)

        MF = np.random.uniform(
            max(avg_mf_target * 0.9, 0.0),
            min(avg_mf_target * 1.1, 1.0),
            size=n_muscles,
        )
        remaining = 1.0 - MF
        split = np.random.uniform(0.0, 1.0, size=n_muscles)
        base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split
        base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split)
        base_env.base_env.unwrapped.muscle_fatigue.MF[:] = MF

        base_env.force_scale = force_scale
        base_env.activation_slowdown = activation_slowdown
        base_env._apply_brady()
        base_env.base_env.unwrapped.sim.data.qpos[:] = 0.0
        base_env.base_env.unwrapped.sim.data.qvel[:] = 0.0
        base_env.base_env.unwrapped.sim.forward()

        return float(np.mean(base_env.base_env.unwrapped.muscle_fatigue.MF))

    def get_exo_obs():
        raw = base_env._current_raw_obs()
        obs_flat = base_env._build_obs(raw)  # includes extra dims when extra_obs=True
        if isinstance(exo_env, TemporalStackWrapper):
            exo_env._buffer.clear()
            for _ in range(exo_env.window):
                exo_env._buffer.append(obs_flat.copy())  # explicit copy — each entry independent
            return exo_env._stack()
        return obs_flat

    # Track 1: Healthy
    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = [target_angle]
    healthy_env.unwrapped.target_type = "fixed"
    healthy_env.unwrapped.update_target(restore_sim=True)
    healthy_env.unwrapped.sim.data.qpos[:] = 0.0
    healthy_env.unwrapped.sim.data.qvel[:] = 0.0
    healthy_env.unwrapped.sim.forward()

    h_obs = healthy_env.unwrapped.get_obs()[: healthy_policy.observation_space.shape[0]]
    healthy_angles = []
    for _ in range(MAX_STEPS):
        action, _ = healthy_policy.predict(h_obs, deterministic=True)
        next_obs, _, done, truncated, _ = healthy_env.step(action)
        h_obs = next_obs[: healthy_policy.observation_space.shape[0]]
        healthy_angles.append(float(healthy_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break

    # Track 2: Impaired, no exo (exo torque = 0)
    exo_env.reset()
    configure_exo_env()
    no_exo_obs = get_exo_obs()
    no_exo_angles = []
    zero_action = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    for _ in range(MAX_STEPS):
        no_exo_obs, _, done, truncated, _ = exo_env.step(zero_action)
        no_exo_angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break

    # Track 3: Impaired + exo
    exo_env.reset()
    actual_avg_mf = configure_exo_env()
    exo_obs = get_exo_obs()

    exo_angles, latencies = [], []
    total_reward = 0.0
    goal_achieved = False
    goal_time = None

    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    for step in range(MAX_STEPS):
        t0 = time.perf_counter()
        if is_recurrent_exo:
            action, lstm_states = exo_policy.predict(
                exo_obs, state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = exo_policy.predict(exo_obs, deterministic=True)
        latencies.append((time.perf_counter() - t0) * 1000)

        exo_obs, reward, done, truncated, _ = exo_env.step(action)
        total_reward += reward
        exo_angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))

        solved_val = base_env.base_env.unwrapped.rwd_dict.get("solved", False)
        solved = bool(np.asarray(solved_val).flat[0])
        if solved and not goal_achieved:
            goal_achieved = True
            goal_time = step * base_env.base_env.unwrapped.dt
        if done or truncated:
            break

    severity = compute_severity(force_scale, activation_slowdown, actual_avg_mf)

    min_len = min(len(healthy_angles), len(exo_angles))
    corr = 0.0
    if min_len > 1:
        corr, _ = pearsonr(healthy_angles[:min_len], exo_angles[:min_len])

    # no-exo vs healthy
    min_len_no = min(len(healthy_angles), len(no_exo_angles))
    no_exo_corr = 0.0
    if min_len_no > 1:
        no_exo_corr, _ = pearsonr(healthy_angles[:min_len_no], no_exo_angles[:min_len_no])

    return {
        "healthy_angles": healthy_angles,
        "no_exo_angles": no_exo_angles,
        "exo_angles": exo_angles,
        "correlation": corr,
        "no_exo_correlation": no_exo_corr,
        "reward": total_reward,
        "goal_achieved": goal_achieved,
        "goal_time": goal_time,
        "latency_ms": latencies,
        "target_angle": target_angle,
        "force_scale": force_scale,
        "activation_slowdown": activation_slowdown,
        "avg_mf": actual_avg_mf,
        "severity": severity,
    }


# -- Plotting --

def plot_joint_angle_overlay(trials: list, angle_edges: np.ndarray, out_dir: str) -> None:
    """One representative trial per angle bin — all 3 tracks overlaid."""
    fig, axes = plt.subplots(1, ANGLE_BINS, figsize=(16, 4), sharey=True)
    for bin_idx, ax in enumerate(axes):
        bin_trials = [t for t in trials if
                      get_angle_bin(t["target_angle"], angle_edges) == bin_idx]
        if not bin_trials:
            ax.set_title(ANGLE_LABELS[bin_idx])
            continue
        t = bin_trials[0]
        ax.plot(range(len(t["healthy_angles"])), t["healthy_angles"],
                label="Healthy", color="steelblue")
        ax.plot(range(len(t["no_exo_angles"])), t["no_exo_angles"],
                label="Impaired, no exo", color="gray", linestyle="--")
        ax.plot(range(len(t["exo_angles"])), t["exo_angles"],
                label="Impaired + exo", color="coral")
        ax.axhline(t["target_angle"], color="black", linestyle=":", linewidth=0.8,
                   label="Target")
        ax.set_title(f"Angle bin: {ANGLE_LABELS[bin_idx]} rad")
        ax.set_xlabel("Step")
        if bin_idx == 0:
            ax.set_ylabel("Joint Angle (rad)")
        if bin_idx == ANGLE_BINS - 1:
            ax.legend(fontsize=7)
    plt.suptitle("Joint Angle Trajectories by Angle Bin")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "joint_angle_overlay.png"), dpi=150)
    plt.close()


def plot_correlation_summary(trials: list, angle_edges: np.ndarray, out_dir: str) -> None:
    mean_corrs = []
    for bin_idx in range(ANGLE_BINS):
        bin_trials = [t for t in trials if
                      get_angle_bin(t["target_angle"], angle_edges) == bin_idx]
        val = np.mean([t["correlation"] for t in bin_trials]) if bin_trials else 0.0
        mean_corrs.append(float(val))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(ANGLE_LABELS, mean_corrs, color="steelblue")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target Angle Bin (rad)")
    ax.set_ylabel("Mean Pearson Correlation (vs Healthy)")
    ax.set_title("Movement Recovery: Impaired+Exo vs Healthy")
    for bar, val in zip(bars, mean_corrs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_summary.png"), dpi=150)
    plt.close()


def plot_reward_per_trial(trials: list, out_dir: str) -> None:
    rewards = [t["reward"] for t in trials]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rewards, marker="o", markersize=4)
    ax.axhline(float(np.mean(rewards)), color="red", linestyle="--",
               label=f"Mean = {np.mean(rewards):.2f}")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward per Trial")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "reward_per_trial.png"), dpi=150)
    plt.close()


def plot_goal_achievement(trials: list, out_dir: str) -> None:
    n = len(trials)
    achieved = sum(t["goal_achieved"] for t in trials)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(["Achieved", "Not Achieved"], [achieved, n - achieved],
           color=["steelblue", "lightgray"])
    ax.set_ylabel("Trials")
    ax.set_title(f"Goal Achievement — {achieved}/{n} ({100*achieved/n:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "goal_achievement.png"), dpi=150)
    plt.close()


# -- Main --

def main():
    parser = argparse.ArgumentParser(
        description="3-track evaluation: healthy / impaired-no-exo / impaired+exo"
    )
    parser.add_argument("--trials", type=int, default=32,
                        help="Total evaluation trials (distributed evenly across matrix)")
    parser.add_argument("--out-dir", default="",
                        help="Output directory (default: results/evaluation or auto-named from policy)")
    parser.add_argument("--extraobs", action="store_true",
                        help="Pass extra_obs=True to env (also auto-detected from policy filename)")
    parser.add_argument("--recurrent", action="store_true",
                        help="Load as RecurrentPPO (PPO-LSTM from sb3_contrib)")
    parser.add_argument("--healthy-path", default="",
                        help="Path to healthy policy (skips interactive prompt)")
    parser.add_argument("--exo-path", default="",
                        help="Path to exo policy to evaluate (skips interactive prompt)")
    args = parser.parse_args()

    print("=" * 60)
    print("reGainX — Policy Evaluation")
    print("=" * 60)
    if args.healthy_path:
        healthy_path = args.healthy_path
    else:
        healthy_path = input("Enter path to healthy policy [policies/healthy_policy.zip]: ").strip()
        if not healthy_path:
            healthy_path = "policies/healthy_policy.zip"
    if args.exo_path:
        exo_path = args.exo_path
    else:
        exo_path = input("Enter path to exo policy to evaluate [policies/policy_brady_deg.zip]: ").strip()
        if not exo_path:
            exo_path = "policies/policy_brady_deg.zip"

    policy_basename = os.path.basename(exo_path).replace(".zip", "")
    is_lstm         = ("lstm" in policy_basename) and ("recurrent" not in policy_basename)
    extra_obs       = args.extraobs or ("extraobs" in policy_basename)
    is_recurrent    = args.recurrent or ("recurrent" in policy_basename)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = f"results/eval_{policy_basename}"

    os.makedirs(out_dir, exist_ok=True)

    print(f"  Policy      : {exo_path}")
    print(f"  CNN-LSTM    : {is_lstm}")
    print(f"  RecurrentPPO: {is_recurrent}")
    print(f"  ExtraObs    : {extra_obs}")
    print(f"  Out dir     : {out_dir}")

    healthy_policy = PPO.load(healthy_path)
    if is_recurrent:
        if not _HAS_RECURRENT_PPO:
            raise ImportError("sb3_contrib not installed — run: pip install sb3-contrib")
        exo_policy = RecurrentPPO.load(exo_path)
    else:
        exo_policy = PPO.load(exo_path)

    healthy_env = gym.make("myoElbowPose1D6MRandom-v0")
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    exo_env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=healthy_path,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=extra_obs,
    )
    if is_lstm:
        exo_env = TemporalStackWrapper(exo_env, window=20)
    # RecurrentPPO handles temporal context internally — no wrapper needed

    # Angle edges from actual env range
    tmp = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    low = float(tmp.unwrapped.target_jnt_range[0, 0])
    high = float(tmp.unwrapped.target_jnt_range[0, 1])
    tmp.close()
    angle_edges = np.linspace(low, high, ANGLE_BINS + 1)

    trial_plan = plan_trials(args.trials)
    print(f"\nRunning {args.trials} trials across {ANGLE_BINS}x{SEVERITY_BINS} matrix cells")
    print(f"  Policy: {exo_path}")

    all_trials = []
    matrix = np.full((ANGLE_BINS, SEVERITY_BINS), np.nan)
    mat_counts = np.zeros((ANGLE_BINS, SEVERITY_BINS), dtype=int)
    mat_sums = np.zeros((ANGLE_BINS, SEVERITY_BINS))
    matrix_no_exo      = np.full((ANGLE_BINS, SEVERITY_BINS), np.nan)
    mat_no_exo_counts  = np.zeros((ANGLE_BINS, SEVERITY_BINS), dtype=int)
    mat_no_exo_sums    = np.zeros((ANGLE_BINS, SEVERITY_BINS))
    severity_edges = np.linspace(0.0, 1.0, SEVERITY_BINS + 1)

    for i, (angle_bin, sev_quartile) in enumerate(trial_plan):
        print(f"  Trial {i+1:3d}/{args.trials}  bin=({angle_bin},{sev_quartile})", end=" ")

        target_angle = angle_bin_to_target(angle_bin, angle_edges)
        fs_range, sl_range, mf_range = severity_quartile_to_range(sev_quartile)
        force_scale = float(np.random.uniform(*fs_range))
        activation_slowdown = float(np.random.uniform(*sl_range))
        avg_mf_target = float(np.random.uniform(*mf_range))

        trial = run_eval_trial(
            exo_env, exo_policy,
            healthy_env, healthy_policy,
            target_angle, force_scale, activation_slowdown, avg_mf_target,
            is_recurrent_exo=is_recurrent,
        )
        all_trials.append(trial)

        mat_sums[angle_bin, sev_quartile] += trial["correlation"]
        mat_counts[angle_bin, sev_quartile] += 1
        if mat_counts[angle_bin, sev_quartile] >= 1:
            matrix[angle_bin, sev_quartile] = (
                mat_sums[angle_bin, sev_quartile] / mat_counts[angle_bin, sev_quartile]
            )

        mat_no_exo_sums[angle_bin, sev_quartile]   += trial["no_exo_correlation"]
        mat_no_exo_counts[angle_bin, sev_quartile] += 1
        if mat_no_exo_counts[angle_bin, sev_quartile] >= 1:
            matrix_no_exo[angle_bin, sev_quartile] = (
                mat_no_exo_sums[angle_bin, sev_quartile]
                / mat_no_exo_counts[angle_bin, sev_quartile]
            )

        print(f"reward={trial['reward']:7.2f}  corr={trial['correlation']:.3f}  "
              f"goal={'Y' if trial['goal_achieved'] else 'N'}")

    # Mark cells with no trials as NaN
    matrix[mat_counts < 1] = np.nan
    matrix_no_exo[mat_no_exo_counts < 1] = np.nan

    print(f"\nGenerating plots -> {out_dir}/")
    plot_joint_angle_overlay(all_trials, angle_edges, out_dir)
    plot_correlation_summary(all_trials, angle_edges, out_dir)
    plot_reward_per_trial(all_trials, out_dir)
    plot_goal_achievement(all_trials, out_dir)
    plot_confusion_matrix(
        matrix, ANGLE_LABELS, SEVERITY_LABELS,
        f"Movement Accuracy (with exo) — {policy_basename}",
        os.path.join(out_dir, "confusion_matrix_with_exo.png"),
        pct=True,
    )
    plot_confusion_matrix(
        matrix_no_exo, ANGLE_LABELS, SEVERITY_LABELS,
        f"Movement Accuracy (no exo) — {policy_basename}",
        os.path.join(out_dir, "confusion_matrix_no_exo.png"),
        pct=True,
    )

    mean_corr = float(np.nanmean([t["correlation"] for t in all_trials]))
    mean_reward = float(np.mean([t["reward"] for t in all_trials]))
    goal_rate = float(np.mean([t["goal_achieved"] for t in all_trials]))
    print(f"\n{'='*60}")
    print(f"Evaluation summary — {policy_basename}")
    print(f"  Policy          : {policy_basename}")
    print(f"  Trials          : {args.trials}")
    print(f"  Mean reward     : {mean_reward:.2f}")
    print(f"  Goal rate       : {goal_rate:.1%}")
    print(f"  Mean Pearson r  : {mean_corr:.3f}  (higher = closer to healthy)")
    print(f"{'='*60}")

    healthy_env.close()
    exo_env.close()


if __name__ == "__main__":
    main()
