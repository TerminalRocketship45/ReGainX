"""
EMG Noise Robustness Evaluation

Tests all policies across 40 noise levels (sigma 0.0 → 0.50).
2 fixed episodes per noise level — shared across ALL policies so every
policy faces identical patient states, enabling fair comparison.

X-axis : EMG noise sigma (std of additive Gaussian noise on muscle activations)
Y-axis : Mean Pearson r vs healthy baseline

Outputs:
  results/noise/noise_robustness.png    — absolute performance, one line per policy
  results/noise/noise_degradation.png   — relative retention (r/r0), shows how
                                          each policy degrades as noise grows
  results/noise/noise_robustness.csv    — raw numbers for further analysis

Usage:
  python noise_eval.py \\
    --healthy    policies/healthy_policy.zip \\
    --brady      policies/policy_brady_deg.zip \\
    --deg        policies/policy_deg.zip \\
    --lstm       policies/policy_brady_deg_lstm.zip \\
    --deg-lstm   policies/policy_deg_lstm.zip \\
    --recurrent  policies/policy_brady_deg_recurrent.zip \\
    --noisy-recurrent policies/policy_brady_deg_recurrent_noisy.zip
"""

import argparse
import csv
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
from envs.elbow_env_noisy import NoisyExoWrapper
from envs.temporal_buffer import TemporalStackWrapper

JOINT_LOW          = 0.0
JOINT_HIGH         = 2.27
MAX_STEPS          = 200
N_NOISE_LEVELS     = 40
SIGMA_MIN          = 0.0    # clean baseline
SIGMA_MAX          = 0.50   # ~0 dB SNR — extreme / pathological noise
EPISODES_PER_LEVEL = 2      # shared episodes (same seed) across all policies
OUT_DIR            = "results/noise"
BASE_SEED          = 2025   # master seed for reproducible episode generation

# Ablation-mode constants (--ablation-noise flag)
SIGMA_MAX_ABLATION   = 1.0    # push to extreme / unrealistic noise
N_ABLATION_LEVELS    = 50     # finer resolution
REALISTIC_SIGMA_LOW  = 0.01   # lower bound of realistic EMG noise (SNR ≈ 34 dB)
REALISTIC_SIGMA_HIGH = 0.10   # upper bound of realistic EMG noise (SNR ≈ 14 dB)


# ---------------------------------------------------------------------------
# Episode seed generation
# ---------------------------------------------------------------------------

def _generate_episode_seeds(n_episodes: int, master_seed: int) -> np.ndarray:
    return np.random.default_rng(master_seed).integers(0, 2**31, size=n_episodes)


def _sample_patient_state(rng: np.random.RandomState, n_muscles: int,
                           force_scale_range: tuple, activation_slowdown_range: tuple):
    """Draw patient parameters deterministically from rng."""
    target_angle    = rng.uniform(JOINT_LOW, JOINT_HIGH)
    MF              = rng.uniform(0.7, 1.0, size=n_muscles)
    remaining       = 1.0 - MF
    split           = rng.uniform(0.0, 1.0, size=n_muscles)
    force_scale     = rng.uniform(*force_scale_range)
    activation_slow = rng.uniform(*activation_slowdown_range)
    return target_angle, MF, remaining, split, force_scale, activation_slow


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _make_noisy_exo_env(healthy_path: str, sigma: float,
                         lstm: bool = False, extra_obs: bool = False):
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    inner = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=healthy_path,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=extra_obs,
    )
    if lstm:
        inner = TemporalStackWrapper(inner, window=20)
    return NoisyExoWrapper(inner, noise_sigma=sigma, randomize_sigma=False)


def _load_policy(path: str, is_recurrent: bool = False):
    if is_recurrent:
        if not _HAS_RECURRENT_PPO:
            raise ImportError("sb3_contrib not installed — run: pip install sb3-contrib")
        return RecurrentPPO.load(path)
    return PPO.load(path)


def _unpack_base_env(noisy_env: NoisyExoWrapper) -> CombinedExoOnlyWrapper:
    """Navigate NoisyExoWrapper → (TemporalStackWrapper →) CombinedExoOnlyWrapper."""
    inner = noisy_env.env
    if isinstance(inner, TemporalStackWrapper):
        return inner.env
    return inner


# ---------------------------------------------------------------------------
# Episode setup: override patient state from seed after env.reset()
# ---------------------------------------------------------------------------

def _configure_env(env: NoisyExoWrapper, seed: int,
                   n_muscles: int, force_scale_range: tuple,
                   activation_slowdown_range: tuple):
    """
    Reset env and install seeded patient state.
    Returns the initial observation (with noise applied) for the exo policy.
    """
    rng = np.random.RandomState(int(seed))
    target_angle, MF, remaining, split, force_scale, activation_slow = \
        _sample_patient_state(rng, n_muscles, force_scale_range, activation_slowdown_range)

    inner   = env.env               # TemporalStackWrapper or CombinedExoOnlyWrapper
    base_env = _unpack_base_env(env) # CombinedExoOnlyWrapper

    env.reset()  # sets _current_sigma, resets inner env (patient state overridden below)

    base_env.base_env.unwrapped.target_jnt_value = [target_angle]
    base_env.base_env.unwrapped.target_type = "fixed"
    base_env.base_env.unwrapped.update_target(restore_sim=True)
    base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split
    base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split)
    base_env.base_env.unwrapped.muscle_fatigue.MF[:] = MF
    base_env.force_scale          = force_scale
    base_env.activation_slowdown  = activation_slow
    base_env._apply_brady()
    base_env.base_env.unwrapped.sim.data.qpos[:] = 0.0
    base_env.base_env.unwrapped.sim.data.qvel[:] = 0.0
    base_env.base_env.unwrapped.sim.forward()

    raw      = base_env._current_raw_obs()
    obs_flat = base_env._build_obs(raw)

    if isinstance(inner, TemporalStackWrapper):
        inner._buffer.clear()
        for _ in range(inner.window):
            inner._buffer.append(obs_flat.copy())
        obs = inner._stack()
    else:
        obs = obs_flat

    return env._inject_noise(obs)


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def _run_healthy_track(healthy_env, healthy_policy: PPO, seed: int) -> list:
    """Run healthy policy to the target angle encoded in seed."""
    rng = np.random.RandomState(int(seed))
    target_angle = rng.uniform(JOINT_LOW, JOINT_HIGH)  # first draw matches _sample_patient_state

    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = [target_angle]
    healthy_env.unwrapped.target_type = "fixed"
    healthy_env.unwrapped.update_target(restore_sim=True)
    healthy_env.unwrapped.sim.data.qpos[:] = 0.0
    healthy_env.unwrapped.sim.data.qvel[:] = 0.0
    healthy_env.unwrapped.sim.forward()

    obs = healthy_env.unwrapped.get_obs()[:healthy_policy.observation_space.shape[0]]
    angles = []
    for _ in range(MAX_STEPS):
        action, _ = healthy_policy.predict(obs, deterministic=True)
        next_obs, _, done, truncated, _ = healthy_env.step(action)
        obs = next_obs[:healthy_policy.observation_space.shape[0]]
        angles.append(float(healthy_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles


def _run_exo_track(env: NoisyExoWrapper, policy, obs: np.ndarray,
                   is_recurrent: bool) -> list:
    """Step exo policy from pre-configured obs. Returns qpos[0] angle list."""
    base_env = _unpack_base_env(env)
    angles        = []
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)

    for _ in range(MAX_STEPS):
        if is_recurrent:
            action, lstm_states = policy.predict(
                obs, state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, _, done, truncated, _ = env.step(action)
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles


def _run_exo_track_with_reward(env: NoisyExoWrapper, policy, obs: np.ndarray,
                               is_recurrent: bool):
    """Like _run_exo_track but also returns cumulative episode reward."""
    base_env      = _unpack_base_env(env)
    angles        = []
    total_reward  = 0.0
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)

    for _ in range(MAX_STEPS):
        if is_recurrent:
            action, lstm_states = policy.predict(
                obs, state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = policy.predict(obs, deterministic=True)

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles, total_reward


def _pearsonr_safe(a: list, b: list) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    r, _ = pearsonr(a[:n], b[:n])
    return float(r) if not np.isnan(r) else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _snr_db_to_sigma(snr_db: float) -> float:
    """Convert SNR (dB) to sigma for a normalised EMG signal (mean amp ≈ 0.5)."""
    return 0.5 / (10 ** (snr_db / 20.0))


_NOISY_LABEL = "RecPPO noisy"   # label used in CONFIGS — highlighted in both plots

# SNR levels shown as vertical reference lines across the extended sigma range
_SNR_REFS = [25, 20, 15, 10, 6]   # dB; 6 dB ≈ sigma 0.25 (pathological noise)


def _snr_annotation(ax, sigmas):
    """Draw vertical SNR reference lines within the plotted sigma range."""
    lo, hi = sigmas[0], sigmas[-1]
    for snr_db in _SNR_REFS:
        s = _snr_db_to_sigma(snr_db)
        if lo < s < hi:
            ax.axvline(s, color="gray", linestyle=":", linewidth=0.8, alpha=0.65)
            ax.text(s + (hi - lo) * 0.004, ax.get_ylim()[0] + 0.01,
                    f"{snr_db} dB", fontsize=7, color="gray",
                    rotation=90, va="bottom")


def _line_style(label: str, idx: int):
    """Return (color, linewidth, linestyle, markersize) for a policy label."""
    if label == _NOISY_LABEL:
        return "crimson", 2.5, "-", 5
    return f"C{idx}", 1.6, "-", 3


def _plot_results(sigmas: np.ndarray, results: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (label, r_series) in enumerate(results.items()):
        color, lw, ls, ms = _line_style(label, i)
        zorder = 3 if label == _NOISY_LABEL else 2
        ax.plot(sigmas, r_series, label=label,
                color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=ms, zorder=zorder)

    ax.set_xlabel("EMG Noise σ  (std of additive Gaussian noise on muscle activations)",
                  fontsize=10)
    ax.set_ylabel("Mean Pearson r  (vs Healthy)", fontsize=10)
    ax.set_title(
        "EMG Noise Robustness — All Policies\n"
        f"{EPISODES_PER_LEVEL} shared episodes per noise level  ·  "
        f"{N_NOISE_LEVELS} levels  ·  σ ∈ [{SIGMA_MIN:.2f}, {SIGMA_MAX:.2f}]",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(SIGMA_MIN, SIGMA_MAX)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    _snr_annotation(ax, sigmas)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[noise_eval] Plot     → {out_path}")


def _plot_degradation(sigmas: np.ndarray, results: dict, out_path: str) -> None:
    """
    Relative performance retention: r(sigma) / r(sigma=0) for each policy.
    All lines start at 1.0 (clean baseline). A flat line means no degradation.
    Highlights whether the noise-trained policy resists degradation better.
    """
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (label, r_series) in enumerate(results.items()):
        r0 = r_series[0]
        if r0 <= 0:
            continue
        retention = [r / r0 for r in r_series]
        color, lw, ls, ms = _line_style(label, i)
        zorder = 3 if label == _NOISY_LABEL else 2
        ax.plot(sigmas, retention, label=label,
                color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=ms, zorder=zorder)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5,
               label="_no degradation")
    ax.set_xlabel("EMG Noise σ  (std of additive Gaussian noise on muscle activations)",
                  fontsize=10)
    ax.set_ylabel("Relative Performance  r(σ) / r(0)", fontsize=10)
    ax.set_title(
        "Policy Degradation Under Increasing EMG Noise\n"
        "1.0 = no degradation  ·  lower = worse  ·  "
        f"RecPPO noisy shown in red",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(SIGMA_MIN, SIGMA_MAX)
    ax.set_ylim(bottom=0, top=1.05)
    ax.grid(True, alpha=0.3)
    _snr_annotation(ax, sigmas)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[noise_eval] Degrad. → {out_path}")


# ---------------------------------------------------------------------------
# Ablation: RecPPO vs RecPPO-noisy noise impact
# ---------------------------------------------------------------------------

def _shade_realistic(ax):
    """Green band marking the realistic EMG noise range across the full y extent."""
    ax.axvspan(
        REALISTIC_SIGMA_LOW, REALISTIC_SIGMA_HIGH,
        alpha=0.12, color="limegreen", zorder=0,
        label=f"Realistic EMG noise  σ ∈ [{REALISTIC_SIGMA_LOW}, {REALISTIC_SIGMA_HIGH}]  "
              f"(SNR ≈ 14–34 dB)",
    )


def _plot_ablation_single(sigmas: np.ndarray, r_series: list, reward_series: list,
                           label: str, out_dir: str) -> None:
    """Two charts for one policy: Pearson r vs sigma and episode reward vs sigma."""
    safe_label = label.replace(" ", "_")
    color      = "crimson" if label == _NOISY_LABEL else "steelblue"

    for values, ylabel, fname_tag in [
        (r_series,      "Mean Pearson r  (vs Healthy)",  "accuracy"),
        (reward_series, "Mean Episode Reward",            "reward"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(sigmas, values, color=color, linewidth=2.0,
                marker="o", markersize=3.5, zorder=3)
        _shade_realistic(ax)
        _snr_annotation(ax, sigmas)
        ax.set_xlabel("EMG Noise σ  (std of additive Gaussian noise on muscle activations)",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(
            f"{label}  —  {ylabel} vs EMG Noise\n"
            f"σ ∈ [0, {SIGMA_MAX_ABLATION:.1f}]  ·  "
            "green shading = realistic real-world noise range",
            fontsize=11,
        )
        ax.set_xlim(0, sigmas[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, f"ablation_{safe_label}_{fname_tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[noise_eval] Ablation  → {path}")


def _plot_ablation_compare(sigmas: np.ndarray, all_data: dict, out_dir: str) -> None:
    """
    Two comparison charts (reward, accuracy) with both policies on one axes each.
    all_data: {label: {"r": [...], "reward": [...]}}
    """
    _COLORS = {"RecPPO brady_deg": "royalblue", _NOISY_LABEL: "crimson"}

    for metric_key, ylabel, fname_tag in [
        ("r",      "Mean Pearson r  (vs Healthy)",  "accuracy"),
        ("reward", "Mean Episode Reward",            "reward"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))

        for i, (label, series) in enumerate(all_data.items()):
            color = _COLORS.get(label, f"C{i}")
            lw    = 2.5 if label == _NOISY_LABEL else 1.8
            ax.plot(sigmas, series[metric_key], label=label,
                    color=color, linewidth=lw, marker="o", markersize=3.5, zorder=3)

        _shade_realistic(ax)
        _snr_annotation(ax, sigmas)
        metric_title = "Accuracy (Pearson r)" if metric_key == "r" else "Episode Reward"
        ax.set_xlabel("EMG Noise σ  (std of additive Gaussian noise on muscle activations)",
                      fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(
            f"{metric_title} — RecPPO vs RecPPO Noise-Trained\n"
            "red = noise-trained  ·  blue = standard  ·  "
            "green shading = realistic noise range",
            fontsize=11,
        )
        ax.set_xlim(0, sigmas[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = os.path.join(out_dir, f"ablation_compare_{fname_tag}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[noise_eval] Compare   → {path}")


def stage_ablation_noise(args) -> None:
    """
    --ablation-noise mode: evaluate RecPPO (standard brady) and RecPPO (noisy-trained)
    across sigma ∈ [0, 1.0] and produce 6 charts + 1 CSV.

    Outputs:
      ablation_RecPPO_brady_deg_accuracy.png   — Pearson r vs sigma, standard policy
      ablation_RecPPO_brady_deg_reward.png     — episode reward vs sigma, standard policy
      ablation_RecPPO_noisy_accuracy.png       — Pearson r vs sigma, noisy policy
      ablation_RecPPO_noisy_reward.png         — episode reward vs sigma, noisy policy
      ablation_compare_accuracy.png            — both policies, Pearson r
      ablation_compare_reward.png              — both policies, episode reward
      ablation_noise.csv                       — raw numbers
    """
    policies_cfg = [
        (args.recurrent,       "RecPPO brady_deg"),
        (args.noisy_recurrent, _NOISY_LABEL),
    ]
    active  = [(p, lbl) for p, lbl in policies_cfg if os.path.exists(p)]
    missing = [(p, lbl) for p, lbl in policies_cfg if not os.path.exists(p)]

    for _, lbl in missing:
        print(f"[noise_eval] --ablation-noise: skipping missing policy: {lbl}")

    if not active:
        print("[noise_eval] --ablation-noise: no recurrent policies found — train first.")
        return

    sigmas   = np.linspace(0.0, SIGMA_MAX_ABLATION, N_ABLATION_LEVELS)
    ep_seeds = _generate_episode_seeds(EPISODES_PER_LEVEL, args.seed)

    print("=" * 60)
    print("[noise_eval] --ablation-noise: RecPPO vs RecPPO-noisy noise impact")
    print(f"  Sigma range  : 0 → {SIGMA_MAX_ABLATION}  ({N_ABLATION_LEVELS} levels)")
    print(f"  Episodes/lvl : {EPISODES_PER_LEVEL}")
    print(f"  Realistic σ  : {REALISTIC_SIGMA_LOW} – {REALISTIC_SIGMA_HIGH}  (shaded on plots)")
    print(f"  Policies     : {[lbl for _, lbl in active]}")
    print("=" * 60)

    # Healthy reference tracks
    healthy_policy = PPO.load(args.healthy)
    healthy_env    = gym.make("myoElbowPose1D6MRandom-v0")
    print("\n[noise_eval] Computing healthy reference tracks...")
    healthy_angles = [_run_healthy_track(healthy_env, healthy_policy, s) for s in ep_seeds]
    healthy_env.close()

    # Probe env for patient-state parameters
    _probe              = _make_noisy_exo_env(args.healthy, sigma=0.0)
    base_probe          = _unpack_base_env(_probe)
    n_muscles           = base_probe.n_muscles
    force_scale_range   = base_probe.force_scale_range
    activation_slow_range = base_probe.activation_slowdown_range
    _probe.close()

    all_data: dict[str, dict] = {}

    for path, label in active:
        print(f"\n[noise_eval] Evaluating: {label}")
        policy          = _load_policy(path, is_recurrent=True)
        r_per_sigma     = []
        reward_per_sigma = []

        for si, sigma in enumerate(sigmas):
            env    = _make_noisy_exo_env(args.healthy, sigma=sigma)
            ep_rs  = []
            ep_rwd = []

            for ep_idx, seed in enumerate(ep_seeds):
                obs = _configure_env(env, seed, n_muscles,
                                     force_scale_range, activation_slow_range)
                angles, total_rwd = _run_exo_track_with_reward(
                    env, policy, obs, is_recurrent=True)
                ep_rs.append(_pearsonr_safe(angles, healthy_angles[ep_idx]))
                ep_rwd.append(total_rwd)

            env.close()
            r_per_sigma.append(float(np.mean(ep_rs)))
            reward_per_sigma.append(float(np.mean(ep_rwd)))
            print(f"  [{si+1:2d}/{N_ABLATION_LEVELS}] σ={sigma:.3f}  "
                  f"r={r_per_sigma[-1]:.4f}  rwd={reward_per_sigma[-1]:.1f}", end="\r")

        print(f"  Done.  r@σ=0={r_per_sigma[0]:.4f}  "
              f"r@σ=1={r_per_sigma[-1]:.4f}   ")
        all_data[label] = {"r": r_per_sigma, "reward": reward_per_sigma}

    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-policy charts (2 per policy)
    for label, series in all_data.items():
        _plot_ablation_single(sigmas, series["r"], series["reward"], label, OUT_DIR)

    # Comparison charts (2 total)
    _plot_ablation_compare(sigmas, all_data, OUT_DIR)

    # CSV
    csv_path = os.path.join(OUT_DIR, "ablation_noise.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["sigma"]
        for lbl in all_data:
            header += [f"{lbl}_pearson_r", f"{lbl}_episode_reward"]
        writer.writerow(header)
        for si, sigma in enumerate(sigmas):
            row = [f"{sigma:.6f}"]
            for lbl in all_data:
                row += [f"{all_data[lbl]['r'][si]:.6f}",
                        f"{all_data[lbl]['reward'][si]:.4f}"]
            writer.writerow(row)
    print(f"[noise_eval] CSV       → {csv_path}")

    print("\n" + "=" * 60)
    print("[noise_eval] --ablation-noise complete.")
    print(f"  results/noise/ablation_*.png  (6 charts)")
    print(f"  results/noise/ablation_noise.csv")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

POLICY_DEFAULTS = {
    "healthy":        "policies/healthy_policy.zip",
    "brady":          "policies/policy_brady_deg.zip",
    "deg":            "policies/policy_deg.zip",
    "lstm":           "policies/policy_brady_deg_lstm.zip",
    "deg_lstm":       "policies/policy_deg_lstm.zip",
    "extraobs":       "policies/policy_brady_deg_lstm_extraobs.zip",
    "recurrent":      "policies/policy_brady_deg_recurrent.zip",
    "deg_recurrent":  "policies/policy_deg_recurrent.zip",
    "noisy_recurrent":"policies/policy_brady_deg_recurrent_noisy.zip",
}


def main():
    parser = argparse.ArgumentParser(description="EMG noise robustness evaluation")
    parser.add_argument("--healthy",
                        default=POLICY_DEFAULTS["healthy"],
                        help=f"Healthy baseline policy (default: {POLICY_DEFAULTS['healthy']})")
    parser.add_argument("--brady",
                        default=POLICY_DEFAULTS["brady"])
    parser.add_argument("--deg",
                        default=POLICY_DEFAULTS["deg"])
    parser.add_argument("--lstm",
                        default=POLICY_DEFAULTS["lstm"])
    parser.add_argument("--deg-lstm",
                        default=POLICY_DEFAULTS["deg_lstm"])
    parser.add_argument("--extraobs",
                        default=POLICY_DEFAULTS["extraobs"])
    parser.add_argument("--recurrent",
                        default=POLICY_DEFAULTS["recurrent"])
    parser.add_argument("--deg-recurrent",
                        default=POLICY_DEFAULTS["deg_recurrent"])
    parser.add_argument("--noisy-recurrent",
                        default=POLICY_DEFAULTS["noisy_recurrent"])
    parser.add_argument("--seed", type=int, default=BASE_SEED,
                        help="Master seed for episode generation")
    parser.add_argument("--ablation-noise", dest="ablation_noise", action="store_true",
                        help=(
                            "Ablation mode: evaluate only RecPPO (standard) and "
                            "RecPPO (noisy-trained) across sigma 0 → 1.0. "
                            "Produces 6 charts + CSV in results/noise/. "
                            "Skips the standard all-policy evaluation."
                        ))
    args = parser.parse_args()

    if args.ablation_noise:
        if not os.path.exists(args.healthy):
            print(f"[noise_eval] ERROR: healthy policy not found at '{args.healthy}'")
            return
        stage_ablation_noise(args)
        return

    if not os.path.exists(args.healthy):
        print(f"[noise_eval] ERROR: healthy policy not found at '{args.healthy}'")
        print("  Train it first or pass --healthy <path>")
        return

    CONFIGS = [
        (args.brady,           False, False, False, "MLP brady_deg"),
        (args.deg,             False, False, False, "MLP deg"),
        (args.lstm,            True,  False, False, "CNN-LSTM brady_deg"),
        (args.deg_lstm,        True,  False, False, "CNN-LSTM deg"),
        (args.extraobs,        True,  True,  False, "CNN-LSTM extraobs"),
        (args.recurrent,       False, False, True,  "RecPPO brady_deg"),
        (args.deg_recurrent,   False, False, True,  "RecPPO deg"),
        (args.noisy_recurrent, False, False, True,  "RecPPO noisy"),
    ]
    active = [(p, lstm, eo, rec, lbl) for (p, lstm, eo, rec, lbl) in CONFIGS
              if os.path.exists(p)]

    if not active:
        print("[noise_eval] No exo policy files found in policies/ — train at least one first.")
        return

    skipped = [(p, lbl) for (p, _, _, _, lbl) in CONFIGS if not os.path.exists(p)]
    if skipped:
        print(f"[noise_eval] Skipping {len(skipped)} missing policies: "
              + ", ".join(lbl for _, lbl in skipped))

    sigmas    = np.linspace(SIGMA_MIN, SIGMA_MAX, N_NOISE_LEVELS)
    ep_seeds  = _generate_episode_seeds(EPISODES_PER_LEVEL, args.seed)

    print("=" * 60)
    print("[noise_eval] EMG Noise Robustness Evaluation")
    print(f"  Noise levels  : {N_NOISE_LEVELS}  (σ {SIGMA_MIN:.3f} → {SIGMA_MAX:.3f})")
    print(f"  Episodes/level: {EPISODES_PER_LEVEL}  (same seeds for all policies)")
    print(f"  Total evals   : {N_NOISE_LEVELS * EPISODES_PER_LEVEL} per policy")
    print(f"  Master seed   : {args.seed}")
    print(f"  Policies      : {len(active)}")
    print("=" * 60)

    # Pre-compute healthy reference tracks (no noise on healthy side)
    healthy_policy = PPO.load(args.healthy)
    healthy_env    = gym.make("myoElbowPose1D6MRandom-v0")
    print("\n[noise_eval] Computing healthy reference tracks...")
    healthy_angles_per_ep = []
    for seed in ep_seeds:
        angles = _run_healthy_track(healthy_env, healthy_policy, seed)
        healthy_angles_per_ep.append(angles)
    healthy_env.close()
    print(f"  Done ({EPISODES_PER_LEVEL} reference tracks)")

    # Probe one env instance for muscle count and impairment ranges
    _probe = _make_noisy_exo_env(args.healthy, sigma=0.0)
    base_probe             = _unpack_base_env(_probe)
    n_muscles              = base_probe.n_muscles
    force_scale_range      = base_probe.force_scale_range
    activation_slow_range  = base_probe.activation_slowdown_range
    _probe.close()

    # Main evaluation loop
    results: dict[str, list] = {}

    for path, lstm, extra_obs, is_recurrent, label in active:
        print(f"\n[noise_eval] Evaluating: {label}")
        policy = _load_policy(path, is_recurrent=is_recurrent)
        mean_r_per_sigma = []

        for si, sigma in enumerate(sigmas):
            env = _make_noisy_exo_env(args.healthy, sigma=sigma,
                                       lstm=lstm, extra_obs=extra_obs)
            ep_rs = []
            for ep_idx, seed in enumerate(ep_seeds):
                obs = _configure_env(env, seed, n_muscles,
                                     force_scale_range, activation_slow_range)
                exo_angles = _run_exo_track(env, policy, obs, is_recurrent)
                r = _pearsonr_safe(exo_angles, healthy_angles_per_ep[ep_idx])
                ep_rs.append(r)
            env.close()

            mean_r = float(np.mean(ep_rs))
            mean_r_per_sigma.append(mean_r)
            print(f"  [{si+1:2d}/{N_NOISE_LEVELS}] σ={sigma:.4f}  r={mean_r:.4f}", end="\r")

        print(f"  Done.  r@σ=0.00: {mean_r_per_sigma[0]:.4f}  "
              f"r@σ={SIGMA_MAX:.2f}: {mean_r_per_sigma[-1]:.4f}   ")
        results[label] = mean_r_per_sigma

    # Save CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "noise_robustness.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sigma"] + list(results.keys()))
        for si, sigma in enumerate(sigmas):
            row = [f"{sigma:.6f}"] + [f"{results[lbl][si]:.6f}" for lbl in results]
            writer.writerow(row)
    print(f"\n[noise_eval] CSV  → {csv_path}")

    # Plots
    plot_path  = os.path.join(OUT_DIR, "noise_robustness.png")
    degrad_path = os.path.join(OUT_DIR, "noise_degradation.png")
    _plot_results(sigmas, results, plot_path)
    _plot_degradation(sigmas, results, degrad_path)

    # Summary table
    print("\n" + "=" * 60)
    print(f"[noise_eval] Summary (r at σ=0 vs σ={SIGMA_MAX:.2f})")
    print(f"  {'Policy':30s}  {'r@0':>7}  {'r@max':>7}  {'drop':>7}  {'retained':>9}")
    print("  " + "-" * 62)
    for label, r_series in results.items():
        r0   = r_series[0]
        rmax = r_series[-1]
        drop = r0 - rmax
        pct  = (rmax / r0 * 100) if r0 > 0 else 0.0
        marker = " ◄" if label == _NOISY_LABEL else ""
        print(f"  {label:30s}  {r0:7.4f}  {rmax:7.4f}  {drop:+7.4f}  {pct:7.1f}%{marker}")
    print("=" * 60)


if __name__ == "__main__":
    main()
