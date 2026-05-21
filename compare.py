"""
Systematic ablation comparison of trained exo policies.

Three ablation studies — all evaluations on brady+deg patients:
  CNN ablation     : policy_brady_deg vs policy_brady_deg_cnn  → results/cnn_ablation/
  Brady ablation   : policy_deg_cnn vs policy_brady_deg_cnn   → results/bradykinesia_ablation/
  ExtraObs ablation: policy_brady_deg vs policy_brady_deg_extraobs → results/extraobs_ablation/

Usage:
    python compare.py

The script prompts for all policy paths interactively (defaults provided).
Noise robustness test is included only in the CNN ablation comparison.
"""

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

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.temporal_buffer import TemporalStackWrapper
from models.temporal_cnn import TemporalCNNExtractor
from utils import (
    compute_severity, get_angle_bin, get_severity_quartile,
    plot_confusion_matrix, save_video, add_text_to_frame,
)

N_TRIALS = 30
MAX_STEPS = 500
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
NOISE_EPISODES_PER_LEVEL = 10
ANGLE_BINS = 4
SEVERITY_BINS = 4

ANGLE_LABELS = ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5"]
SEVERITY_LABELS = ["Q1 mild", "Q2", "Q3", "Q4 severe"]


# -- Environment builders --

def make_healthy_env():
    return gym.make("myoElbowPose1D6MRandom-v0")


def make_exo_env(healthy_path: str, cnn: bool = False, extra_obs: bool = False):
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=healthy_path,
        bradykinesia=True,  # always evaluate on brady+deg
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=extra_obs,
    )
    if cnn:
        env = TemporalStackWrapper(env, window=20)
    return env


# -- Trial runner --

def run_trial(
    exo_env,
    exo_policy: PPO,
    healthy_env,
    healthy_policy: PPO,
    seed: int,
    angle_edges: np.ndarray,
) -> dict:
    """
    Run one evaluation trial with shared seed for fair patient comparison.
    Both exo_env and healthy_env face the same target angle and patient state.
    """
    rng = np.random.RandomState(seed)

    base_env = exo_env.env if isinstance(exo_env, TemporalStackWrapper) else exo_env
    low = float(base_env.base_env.unwrapped.target_jnt_range[0, 0])
    high = float(base_env.base_env.unwrapped.target_jnt_range[0, 1])
    target_angle = rng.uniform(low, high)

    n_muscles = base_env.n_muscles
    MF = rng.uniform(0.7, 1.0, size=n_muscles)
    remaining = 1.0 - MF
    split = rng.uniform(0.0, 1.0, size=n_muscles)

    force_scale = rng.uniform(*base_env.force_scale_range)
    activation_slowdown = rng.uniform(*base_env.activation_slowdown_range)

    # -- Set up healthy env --
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

    # -- Set up exo env with seeded patient state --
    exo_obs, _ = exo_env.reset()
    base_env.base_env.unwrapped.target_jnt_value = [target_angle]
    base_env.base_env.unwrapped.target_type = "fixed"
    base_env.base_env.unwrapped.update_target(restore_sim=True)
    base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split
    base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split)
    base_env.base_env.unwrapped.muscle_fatigue.MF[:] = MF
    base_env.force_scale = force_scale
    base_env.activation_slowdown = activation_slowdown
    base_env._apply_brady()
    base_env.base_env.unwrapped.sim.data.qpos[:] = 0.0
    base_env.base_env.unwrapped.sim.data.qvel[:] = 0.0
    base_env.base_env.unwrapped.sim.forward()

    # Rebuild obs after seeded patient setup (ensures force_scale/activation_slowdown are current)
    if isinstance(exo_env, TemporalStackWrapper):
        raw = base_env._current_raw_obs()
        exo_obs_flat = base_env._exo_obs(raw)
        exo_env._buffer.clear()
        for _ in range(exo_env.window):
            exo_env._buffer.append(exo_obs_flat.astype(np.float32, copy=False))
        exo_obs = exo_env._stack()
    else:
        exo_obs = base_env._build_obs(base_env._current_raw_obs())

    exo_angles, latencies = [], []
    total_reward = 0.0
    goal_achieved = False
    goal_time = None

    for step in range(MAX_STEPS):
        t0 = time.perf_counter()
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

    avg_mf = float(np.mean(base_env.base_env.unwrapped.muscle_fatigue.MF))
    severity = compute_severity(force_scale, activation_slowdown, avg_mf)

    min_len = min(len(healthy_angles), len(exo_angles))
    corr = 0.0
    if min_len > 1:
        corr, _ = pearsonr(healthy_angles[:min_len], exo_angles[:min_len])

    angle_bin = get_angle_bin(target_angle, angle_edges)

    return {
        "reward": total_reward,
        "goal_achieved": goal_achieved,
        "goal_time": goal_time,
        "latency_ms": latencies,
        "healthy_angles": healthy_angles,
        "exo_angles": exo_angles,
        "correlation": corr,
        "target_angle": target_angle,
        "angle_bin": angle_bin,
        "force_scale": force_scale,
        "activation_slowdown": activation_slowdown,
        "avg_mf": avg_mf,
        "severity": severity,
    }


# -- Confusion matrix builder --

def build_matrix(trials: list, severity_edges: np.ndarray) -> np.ndarray:
    matrix = np.full((ANGLE_BINS, SEVERITY_BINS), np.nan)
    counts = np.zeros((ANGLE_BINS, SEVERITY_BINS), dtype=int)
    sums = np.zeros((ANGLE_BINS, SEVERITY_BINS))

    for t in trials:
        row = t["angle_bin"]
        col = get_severity_quartile(t["severity"], severity_edges)
        sums[row, col] += t["correlation"]
        counts[row, col] += 1

    mask = counts >= 2
    matrix[mask] = sums[mask] / counts[mask]
    return matrix


# -- Plotting helpers --

def plot_reward(trials_a: list, trials_b: list, labels: tuple, save_path: str) -> None:
    rewards_a = [t["reward"] for t in trials_a]
    rewards_b = [t["reward"] for t in trials_b]
    x = np.arange(N_TRIALS)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, rewards_a, label=labels[0], marker="o", markersize=3)
    ax.plot(x, rewards_b, label=labels[1], marker="s", markersize=3)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Reward per Trial")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_temporal(trials_a: list, trials_b: list, labels: tuple, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (trials, lbl) in enumerate(zip([trials_a, trials_b], labels)):
        times = [t["goal_time"] if t["goal_time"] is not None else float("nan")
                 for t in trials]
        ax.scatter(range(N_TRIALS), times, label=lbl, marker="o" if i == 0 else "s", s=20)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Time to First Goal (s) — NaN = not achieved")
    ax.set_title("Time to Goal per Trial")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_goal_achievement(trials_a: list, trials_b: list, labels: tuple, save_path: str) -> None:
    counts = [sum(t["goal_achieved"] for t in trials_a),
              sum(t["goal_achieved"] for t in trials_b)]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, counts, color=["steelblue", "coral"])
    ax.set_ylabel(f"Goals Achieved / {N_TRIALS}")
    ax.set_title("Goal Achievement")
    ax.set_ylim(0, N_TRIALS)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.3,
                str(count), ha="center", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_latency(trials_a: list, trials_b: list, labels: tuple, save_path: str) -> None:
    lat_a = [ms for t in trials_a for ms in t["latency_ms"]]
    lat_b = [ms for t in trials_b for ms in t["latency_ms"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot([lat_a, lat_b], labels=labels, patch_artist=True,
               boxprops=dict(facecolor="lightblue"))
    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title("Per-Step Inference Latency")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def rng_noise(obs: np.ndarray, sigma: float) -> np.ndarray:
    if sigma == 0.0:
        return np.zeros_like(obs)
    return np.random.normal(0, sigma, size=obs.shape).astype(obs.dtype)


def plot_noise_robustness(env_a, policy_a: PPO, env_b, policy_b: PPO,
                          labels: tuple, save_path: str) -> None:
    results = {labels[0]: [], labels[1]: []}

    for sigma in NOISE_LEVELS:
        for policy, env, lbl in [(policy_a, env_a, labels[0]),
                                  (policy_b, env_b, labels[1])]:
            ep_rewards = []
            for _ in range(NOISE_EPISODES_PER_LEVEL):
                obs, _ = env.reset()
                total = 0.0
                for _ in range(MAX_STEPS):
                    noisy_obs = obs + rng_noise(obs, sigma)
                    action, _ = policy.predict(noisy_obs, deterministic=True)
                    obs, reward, done, truncated, _ = env.step(action)
                    total += reward
                    if done or truncated:
                        break
                ep_rewards.append(total)
            results[lbl].append(np.mean(ep_rewards))

    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, values in results.items():
        ax.plot(NOISE_LEVELS, values, marker="o", label=lbl)
    ax.set_xlabel("Observation Noise sigma (Gaussian)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Noise Robustness: CNN vs No-CNN")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# -- Comparison runner --

def run_comparison(
    label: str,
    policy_path_a: str,
    policy_path_b: str,
    healthy_path: str,
    cnn_a: bool,
    cnn_b: bool,
    out_dir: str,
    angle_edges: np.ndarray,
    extra_obs_a: bool = False,
    extra_obs_b: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    label_a = os.path.basename(policy_path_a).replace(".zip", "")
    label_b = os.path.basename(policy_path_b).replace(".zip", "")
    labels = (label_a, label_b)

    print(f"\n{'='*60}")
    print(f"Running {label}: {label_a}  vs  {label_b}")
    print(f"  {N_TRIALS} trials — brady+deg patients — shared seeds per trial")
    print(f"{'='*60}")

    policy_a = PPO.load(policy_path_a)
    policy_b = PPO.load(policy_path_b)
    healthy_policy = PPO.load(healthy_path)

    env_a = make_exo_env(healthy_path, cnn=cnn_a, extra_obs=extra_obs_a)
    env_b = make_exo_env(healthy_path, cnn=cnn_b, extra_obs=extra_obs_b)
    h_env_a = make_healthy_env()
    h_env_b = make_healthy_env()

    seeds = [int(np.random.randint(0, 2**31)) for _ in range(N_TRIALS)]
    trials_a, trials_b = [], []

    for i, seed in enumerate(seeds):
        print(f"  Trial {i+1:2d}/{N_TRIALS} ...", end=" ", flush=True)
        t_a = run_trial(env_a, policy_a, h_env_a, healthy_policy, seed, angle_edges)
        t_b = run_trial(env_b, policy_b, h_env_b, healthy_policy, seed, angle_edges)
        trials_a.append(t_a)
        trials_b.append(t_b)
        print(f"reward_a={t_a['reward']:.1f}  reward_b={t_b['reward']:.1f}")

    all_sev = [t["severity"] for t in trials_a + trials_b]
    severity_edges = np.percentile(all_sev, [0, 25, 50, 75, 100])

    plot_reward(trials_a, trials_b, labels, f"{out_dir}/reward_comparison.png")
    plot_temporal(trials_a, trials_b, labels, f"{out_dir}/temporal_difference.png")
    plot_goal_achievement(trials_a, trials_b, labels, f"{out_dir}/goal_achievement.png")
    plot_latency(trials_a, trials_b, labels, f"{out_dir}/latency_comparison.png")

    matrix_a = build_matrix(trials_a, severity_edges)
    matrix_b = build_matrix(trials_b, severity_edges)
    plot_confusion_matrix(matrix_a, ANGLE_LABELS, SEVERITY_LABELS,
                          f"Movement Accuracy — {label_a}",
                          f"{out_dir}/confusion_matrix_{label_a}.png")
    plot_confusion_matrix(matrix_b, ANGLE_LABELS, SEVERITY_LABELS,
                          f"Movement Accuracy — {label_b}",
                          f"{out_dir}/confusion_matrix_{label_b}.png")

    print(f"\n  Results saved to {out_dir}/")
    env_a.close(); env_b.close(); h_env_a.close(); h_env_b.close()
    return policy_a, policy_b


def _prompt(prompt_text: str, default: str) -> str:
    val = input(f"{prompt_text} [{default}]: ").strip()
    return val if val else default


def main():
    import argparse as _ap
    parser = _ap.ArgumentParser(description="reGainX ablation comparisons")
    parser.add_argument("--healthy",      default="", help="Path to healthy policy")
    parser.add_argument("--brady",        default="", help="Path to policy_brady_deg")
    parser.add_argument("--cnn",          default="", help="Path to policy_brady_deg_cnn")
    parser.add_argument("--deg-cnn",      default="", help="Path to policy_deg_cnn")
    parser.add_argument("--extraobs-pol", default="", help="Path to policy_brady_deg_extraobs")
    cli = parser.parse_args()

    print("=" * 60)
    print("reGainX — Ablation Comparison")
    print("=" * 60)
    print("Three ablation studies — all evaluated on brady+deg patients.\n")

    healthy_path  = cli.healthy      or _prompt("Healthy policy path",              "policies/healthy_policy.zip")
    path_brady    = cli.brady        or _prompt("brady+deg (no CNN) policy path",   "policies/policy_brady_deg.zip")
    path_cnn      = cli.cnn         or _prompt("brady+deg + CNN policy path",       "policies/policy_brady_deg_cnn.zip")
    path_deg_cnn  = cli.deg_cnn     or _prompt("deg-only + CNN policy path",        "policies/policy_deg_cnn.zip")
    path_extraobs = cli.extraobs_pol or _prompt("brady+deg + extraobs policy path", "policies/policy_brady_deg_extraobs.zip")

    # Verify required files exist
    missing = [p for p in [healthy_path, path_brady, path_cnn, path_extraobs]
               if not os.path.exists(p)]
    if missing:
        print("\nERROR — missing policy files:")
        for m in missing:
            print(f"  {m}")
        return

    tmp_env = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    low  = float(tmp_env.unwrapped.target_jnt_range[0, 0])
    high = float(tmp_env.unwrapped.target_jnt_range[0, 1])
    tmp_env.close()
    angle_edges = np.linspace(low, high, ANGLE_BINS + 1)

    # ── CNN Ablation: brady (no CNN) vs brady (CNN) ────────────────────────
    policy_brady, policy_cnn = run_comparison(
        "CNN Ablation",
        path_brady, path_cnn,
        healthy_path,
        cnn_a=False, cnn_b=True,
        out_dir="results/cnn_ablation",
        angle_edges=angle_edges,
    )

    print("\nRunning noise robustness test for CNN ablation...")
    env_no_cnn = make_exo_env(healthy_path, cnn=False)
    env_cnn    = make_exo_env(healthy_path, cnn=True)
    plot_noise_robustness(
        env_no_cnn, policy_brady,
        env_cnn,    policy_cnn,
        labels=(os.path.basename(path_brady).replace(".zip", ""),
                os.path.basename(path_cnn).replace(".zip", "")),
        save_path="results/cnn_ablation/noise_robustness.png",
    )
    env_no_cnn.close(); env_cnn.close()

    # ── Bradykinesia Ablation: deg_cnn vs brady_deg_cnn ───────────────────
    if os.path.exists(path_deg_cnn):
        run_comparison(
            "Bradykinesia Ablation",
            path_deg_cnn, path_cnn,
            healthy_path,
            cnn_a=True, cnn_b=True,
            out_dir="results/bradykinesia_ablation",
            angle_edges=angle_edges,
        )
    else:
        print(f"\nSkipping bradykinesia ablation — {path_deg_cnn} not found.")
        print("  Train with: python train_exo.py --no-bradykinesia --cnn")

    # ── ExtraObs Ablation: brady (no extra) vs brady (extra) ──────────────
    run_comparison(
        "ExtraObs Ablation",
        path_brady, path_extraobs,
        healthy_path,
        cnn_a=False, cnn_b=False,
        out_dir="results/extraobs_ablation",
        angle_edges=angle_edges,
        extra_obs_a=False, extra_obs_b=True,
    )

    print("\n" + "=" * 60)
    print("All ablation comparisons complete.")
    print("  results/cnn_ablation/")
    print("  results/bradykinesia_ablation/  (if policy_deg_cnn.zip available)")
    print("  results/extraobs_ablation/")
    print("=" * 60)


if __name__ == "__main__":
    main()
