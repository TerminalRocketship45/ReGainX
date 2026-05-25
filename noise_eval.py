"""
EMG Noise Robustness Evaluation

Tests all policies across 30 noise levels (sigma 0.0 → 0.15).
2 fixed episodes per noise level — shared across ALL policies so every
policy faces identical patient states, enabling fair comparison.

X-axis : EMG noise sigma (std of additive Gaussian noise on muscle activations)
Y-axis : Mean Pearson r vs healthy baseline

Outputs:
  results/noise/noise_robustness.png   — line plot, one line per policy
  results/noise/noise_robustness.csv   — raw numbers for further analysis

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
N_NOISE_LEVELS     = 30
SIGMA_MIN          = 0.0    # clean baseline
SIGMA_MAX          = 0.15   # ~13 dB SNR — severely noisy surface EMG
EPISODES_PER_LEVEL = 2      # shared episodes (same seed) across all policies
OUT_DIR            = "results/noise"
BASE_SEED          = 2025   # master seed for reproducible episode generation


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


def _plot_results(sigmas: np.ndarray, results: dict, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (label, r_series) in enumerate(results.items()):
        ax.plot(sigmas, r_series, label=label,
                color=f"C{i}", linewidth=1.8, marker="o", markersize=3.5)

    # SNR reference lines
    for snr_db in [20, 15, 10]:
        sigma_ref = _snr_db_to_sigma(snr_db)
        if SIGMA_MIN < sigma_ref < SIGMA_MAX:
            ax.axvline(sigma_ref, color="gray", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(sigma_ref + 0.001, 0.02, f"SNR {snr_db} dB",
                    fontsize=7, color="gray", rotation=90, va="bottom")

    ax.set_xlabel("EMG Noise σ (std of additive Gaussian noise on muscle activations)",
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
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[noise_eval] Plot → {out_path}")


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
    args = parser.parse_args()

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

    # Plot
    plot_path = os.path.join(OUT_DIR, "noise_robustness.png")
    _plot_results(sigmas, results, plot_path)

    # Summary table
    print("\n" + "=" * 60)
    print("[noise_eval] Summary (r at σ=0 vs σ=max)")
    for label, r_series in results.items():
        drop = r_series[0] - r_series[-1]
        print(f"  {label:30s}  r@0={r_series[0]:.4f}  "
              f"r@{SIGMA_MAX:.2f}={r_series[-1]:.4f}  "
              f"drop={drop:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
