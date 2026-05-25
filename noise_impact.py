"""
Noise impact study + RecurPPO vs RecurPPO-noisy comparisons.

Produces up to 5 plots in results/noise/:

  Per-policy noise impact (clean vs noisy, same patient states, 15 episodes):
    noise_impact_mlp_brady.png
    noise_impact_recppo_brady.png
    noise_impact_recppo_noisy.png   (only if policy exists)

  Cross-policy studies (RecurPPO vs RecurPPO-noisy, same 15 patient states):
    study_clean_env.png   — both policies in a noise-FREE environment
    study_noisy_env.png   — both policies in a noise environment (σ=0.05)

Usage:
    python noise_impact.py          # uses preset policy paths, no flags needed
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import myosuite  # noqa
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLICY_DEFAULTS = {
    "healthy":         "policies/healthy_policy.zip",
    "mlp_brady":       "policies/policy_brady_deg.zip",
    "recurrent":       "policies/policy_brady_deg_recurrent.zip",
    "noisy_recurrent": "policies/policy_brady_deg_recurrent_noisy.zip",
}

N_EPISODES   = 15
SIGMA_IMPACT = np.linspace(0.0, 0.15, N_EPISODES)  # per-episode noise, increases across episodes
SIGMA_STUDY  = 0.05                                  # fixed noise level for study B (~20 dB SNR)
MASTER_SEED  = 2025
OUT_DIR      = "results/noise"
JOINT_LOW    = 0.0
JOINT_HIGH   = 2.27
MAX_STEPS    = 200


# ---------------------------------------------------------------------------
# Shared patient-state helpers (same pattern as algo_compare / noise_eval)
# ---------------------------------------------------------------------------

def _episode_seeds(master_seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(master_seed).integers(0, 2**31, size=n)


def _sample_patient_state(rng, n_muscles, force_scale_range, activation_slowdown_range):
    target_angle    = rng.uniform(JOINT_LOW, JOINT_HIGH)
    MF              = rng.uniform(0.7, 1.0, size=n_muscles)
    remaining       = 1.0 - MF
    split           = rng.uniform(0.0, 1.0, size=n_muscles)
    force_scale     = rng.uniform(*force_scale_range)
    activation_slow = rng.uniform(*activation_slowdown_range)
    return target_angle, MF, remaining, split, force_scale, activation_slow


def _unpack_base_env(noisy_env: NoisyExoWrapper) -> CombinedExoOnlyWrapper:
    inner = noisy_env.env
    if isinstance(inner, TemporalStackWrapper):
        return inner.env
    return inner


def _configure_env(env: NoisyExoWrapper, seed: int,
                   n_muscles, force_scale_range, activation_slow_range):
    """Reset and install seeded patient state. Returns initial noised obs."""
    rng = np.random.RandomState(int(seed))
    target_angle, MF, remaining, split, force_scale, activation_slow = \
        _sample_patient_state(rng, n_muscles, force_scale_range, activation_slow_range)

    inner    = env.env
    base_env = _unpack_base_env(env)

    env.reset()

    base_env.base_env.unwrapped.target_jnt_value = [target_angle]
    base_env.base_env.unwrapped.target_type = "fixed"
    base_env.base_env.unwrapped.update_target(restore_sim=True)
    base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split
    base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split)
    base_env.base_env.unwrapped.muscle_fatigue.MF[:] = MF
    base_env.force_scale         = force_scale
    base_env.activation_slowdown = activation_slow
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


def _run_healthy_track(healthy_env, healthy_policy: PPO, seed: int) -> list:
    rng = np.random.RandomState(int(seed))
    target_angle = rng.uniform(JOINT_LOW, JOINT_HIGH)

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
# Core evaluation: run a policy across (seed, sigma) pairs
# ---------------------------------------------------------------------------

def _make_env(healthy_path: str, sigma: float,
              lstm: bool = False, extra_obs: bool = False) -> NoisyExoWrapper:
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    inner = CombinedExoOnlyWrapper(
        base, frozen_policy_path=healthy_path,
        bradykinesia=True, smart_reset=True,
        hide_pose_err=True, extra_obs=extra_obs,
    )
    if lstm:
        inner = TemporalStackWrapper(inner, window=20)
    return NoisyExoWrapper(inner, noise_sigma=sigma, randomize_sigma=False)


def run_episodes(policy, healthy_path: str, lstm: bool, extra_obs: bool,
                 is_recurrent: bool, episode_seeds: np.ndarray,
                 sigmas: np.ndarray, n_muscles: int,
                 force_scale_range: tuple, activation_slow_range: tuple,
                 healthy_angles_per_ep: list, tag: str = "") -> list:
    """
    Run `len(episode_seeds)` episodes, each with its own sigma and patient seed.
    Returns per-episode Pearson r vs healthy.
    """
    results = []
    for i, (seed, sigma) in enumerate(zip(episode_seeds, sigmas)):
        env = _make_env(healthy_path, sigma=sigma, lstm=lstm, extra_obs=extra_obs)
        obs = _configure_env(env, seed, n_muscles, force_scale_range, activation_slow_range)
        angles = _run_exo_track(env, policy, obs, is_recurrent)
        r = _pearsonr_safe(angles, healthy_angles_per_ep[i])
        results.append(r)
        env.close()
        lbl = f" [{tag}]" if tag else ""
        print(f"  ep {i+1:2d}/{len(episode_seeds)}{lbl}  σ={sigma:.4f}  r={r:.4f}", end="\r")
    print()
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_noise_impact(sigmas: np.ndarray, r_clean: list, r_noisy: list,
                       policy_label: str, out_path: str) -> None:
    """One graph: clean vs noisy for a single policy as sigma increases."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(sigmas, r_clean, label="Clean (σ = 0)", color="steelblue",
            linewidth=2, marker="o", markersize=5)
    ax.plot(sigmas, r_noisy, label="With noise (σ increases per episode)",
            color="coral", linewidth=2, marker="o", markersize=5)
    ax.fill_between(sigmas, r_clean, r_noisy, alpha=0.15, color="orange",
                    label="Performance gap")

    # Mean annotations
    ax.axhline(np.mean(r_clean), color="steelblue", linestyle="--",
               linewidth=1, alpha=0.6)
    ax.axhline(np.mean(r_noisy), color="coral", linestyle="--",
               linewidth=1, alpha=0.6)

    # SNR reference lines
    for snr_db in [20, 15, 10]:
        sigma_ref = 0.5 / (10 ** (snr_db / 20.0))
        if sigmas[0] < sigma_ref < sigmas[-1]:
            ax.axvline(sigma_ref, color="gray", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.text(sigma_ref + 0.001, ax.get_ylim()[0] if ax.get_ylim() else 0,
                    f"{snr_db} dB", fontsize=7, color="gray", va="bottom")

    ax.set_xlabel("EMG Noise σ (additive Gaussian, muscle activations only)", fontsize=10)
    ax.set_ylabel("Pearson r  (vs Healthy baseline)", fontsize=10)
    ax.set_title(
        f"Noise Impact: {policy_label}\n"
        f"Clean (σ=0) vs. Noisy (σ 0→0.15) — same patient states, {N_EPISODES} episodes",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(sigmas[0], sigmas[-1])
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Secondary x-axis showing episode number
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(sigmas[::3])
    ax2.set_xticklabels([f"ep {i+1}" for i in range(0, N_EPISODES, 3)], fontsize=7)
    ax2.set_xlabel("Episode", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [noise_impact] Saved → {out_path}")


def _plot_study(episode_nums: range, r_pol1: list, r_pol2: list,
                label1: str, label2: str, title: str,
                sigma_info: str, out_path: str) -> None:
    """Two-policy comparison: per-episode Pearson r at a fixed sigma."""
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(episode_nums, r_pol1, label=label1, color="C0",
            linewidth=2, marker="o", markersize=5)
    ax.plot(episode_nums, r_pol2, label=label2, color="C1",
            linewidth=2, marker="o", markersize=5)

    m1 = float(np.mean(r_pol1))
    m2 = float(np.mean(r_pol2))
    ax.axhline(m1, color="C0", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axhline(m2, color="C1", linestyle="--", linewidth=1.2, alpha=0.6)

    # Mean labels on right margin
    xmax = max(episode_nums) + 0.4
    ax.text(xmax, m1, f"μ={m1:.3f}", color="C0", va="center", fontsize=8,
            fontweight="bold")
    ax.text(xmax, m2, f"μ={m2:.3f}", color="C1", va="center", fontsize=8,
            fontweight="bold")

    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Pearson r  (vs Healthy baseline)", fontsize=10)
    ax.set_title(f"{title}\n{sigma_info} — same patient states, {N_EPISODES} episodes",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(min(episode_nums) - 0.2, xmax + 0.8)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [noise_impact] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    healthy_path  = POLICY_DEFAULTS["healthy"]
    path_mlp      = POLICY_DEFAULTS["mlp_brady"]
    path_recur    = POLICY_DEFAULTS["recurrent"]
    path_noisy    = POLICY_DEFAULTS["noisy_recurrent"]

    if not os.path.exists(healthy_path):
        print(f"[noise_impact] ERROR: healthy policy not found at '{healthy_path}'")
        return

    has_mlp   = os.path.exists(path_mlp)
    has_recur = os.path.exists(path_recur)
    has_noisy = os.path.exists(path_noisy) and _HAS_RECURRENT_PPO

    if not (has_mlp or has_recur):
        print("[noise_impact] No exo policies found — train first.")
        return

    print("=" * 60)
    print("[noise_impact] Noise Impact Study")
    print(f"  Episodes     : {N_EPISODES}")
    print(f"  Noise range  : σ 0.000 → {SIGMA_IMPACT[-1]:.3f} (impact graphs)")
    print(f"  Study σ      : {SIGMA_STUDY} (~20 dB SNR, study_noisy_env)")
    print(f"  MLP brady    : {'found' if has_mlp   else 'MISSING — skip'}")
    print(f"  RecurPPO     : {'found' if has_recur else 'MISSING — skip'}")
    print(f"  RecurPPO noisy: {'found' if has_noisy else 'MISSING — skip'}")
    print("=" * 60)

    seeds = _episode_seeds(MASTER_SEED, N_EPISODES)

    # Healthy reference tracks
    print("\n[noise_impact] Computing healthy reference tracks...")
    healthy_policy = PPO.load(healthy_path)
    healthy_env    = gym.make("myoElbowPose1D6MRandom-v0")
    healthy_per_ep = [_run_healthy_track(healthy_env, healthy_policy, s) for s in seeds]
    healthy_env.close()
    print(f"  Done ({N_EPISODES} tracks)")

    # Probe env for muscle count and impairment ranges
    _probe = _make_env(healthy_path, sigma=0.0)
    bp = _unpack_base_env(_probe)
    n_muscles, fsr, asr = bp.n_muscles, bp.force_scale_range, bp.activation_slowdown_range
    _probe.close()

    # Storage
    clean_sigma  = np.zeros(N_EPISODES)          # all episodes at sigma=0
    study_sigmas = np.full(N_EPISODES, SIGMA_STUDY)

    results = {}  # key -> {"clean": [...], "noisy": [...], "study_noisy": [...]}

    # ── MLP brady ──────────────────────────────────────────────────────────
    if has_mlp:
        print("\n[noise_impact] MLP brady_deg")
        policy = PPO.load(path_mlp)
        print("  → clean run (σ=0)")
        r_clean = run_episodes(policy, healthy_path, False, False, False,
                               seeds, clean_sigma, n_muscles, fsr, asr, healthy_per_ep,
                               tag="clean")
        print("  → noisy run (σ increases)")
        r_noisy = run_episodes(policy, healthy_path, False, False, False,
                               seeds, SIGMA_IMPACT, n_muscles, fsr, asr, healthy_per_ep,
                               tag="noisy")
        results["mlp_brady"] = {"clean": r_clean, "noisy": r_noisy}

    # ── RecurPPO brady ─────────────────────────────────────────────────────
    if has_recur:
        if not _HAS_RECURRENT_PPO:
            print("[noise_impact] sb3_contrib not installed — skipping RecurPPO")
        else:
            print("\n[noise_impact] RecurPPO brady_deg")
            policy = RecurrentPPO.load(path_recur)
            print("  → clean run (σ=0)")
            r_clean = run_episodes(policy, healthy_path, False, False, True,
                                   seeds, clean_sigma, n_muscles, fsr, asr, healthy_per_ep,
                                   tag="clean")
            print("  → noisy run (σ increases)")
            r_noisy = run_episodes(policy, healthy_path, False, False, True,
                                   seeds, SIGMA_IMPACT, n_muscles, fsr, asr, healthy_per_ep,
                                   tag="noisy")
            print("  → study noisy run (σ=0.05 fixed)")
            r_study = run_episodes(policy, healthy_path, False, False, True,
                                   seeds, study_sigmas, n_muscles, fsr, asr, healthy_per_ep,
                                   tag="study")
            results["recurrent"] = {"clean": r_clean, "noisy": r_noisy, "study_noisy": r_study}

    # ── RecurPPO noisy ─────────────────────────────────────────────────────
    if has_noisy:
        print("\n[noise_impact] RecurPPO brady_deg (noise-trained)")
        policy = RecurrentPPO.load(path_noisy)
        print("  → clean run (σ=0)")
        r_clean = run_episodes(policy, healthy_path, False, False, True,
                               seeds, clean_sigma, n_muscles, fsr, asr, healthy_per_ep,
                               tag="clean")
        print("  → noisy run (σ increases)")
        r_noisy = run_episodes(policy, healthy_path, False, False, True,
                               seeds, SIGMA_IMPACT, n_muscles, fsr, asr, healthy_per_ep,
                               tag="noisy")
        print("  → study noisy run (σ=0.05 fixed)")
        r_study = run_episodes(policy, healthy_path, False, False, True,
                               seeds, study_sigmas, n_muscles, fsr, asr, healthy_per_ep,
                               tag="study")
        results["noisy_recurrent"] = {"clean": r_clean, "noisy": r_noisy, "study_noisy": r_study}

    os.makedirs(OUT_DIR, exist_ok=True)
    eps = range(1, N_EPISODES + 1)

    # ── Noise impact graphs ────────────────────────────────────────────────
    print("\n[noise_impact] Generating noise impact graphs...")

    if "mlp_brady" in results:
        _plot_noise_impact(
            SIGMA_IMPACT,
            results["mlp_brady"]["clean"],
            results["mlp_brady"]["noisy"],
            "MLP brady_deg",
            os.path.join(OUT_DIR, "noise_impact_mlp_brady.png"),
        )

    if "recurrent" in results:
        _plot_noise_impact(
            SIGMA_IMPACT,
            results["recurrent"]["clean"],
            results["recurrent"]["noisy"],
            "RecurrentPPO brady_deg",
            os.path.join(OUT_DIR, "noise_impact_recppo_brady.png"),
        )

    if "noisy_recurrent" in results:
        _plot_noise_impact(
            SIGMA_IMPACT,
            results["noisy_recurrent"]["clean"],
            results["noisy_recurrent"]["noisy"],
            "RecurrentPPO brady_deg (noise-trained)",
            os.path.join(OUT_DIR, "noise_impact_recppo_noisy.png"),
        )

    # ── Study graphs (only if both recurrent policies exist) ───────────────
    if "recurrent" in results and "noisy_recurrent" in results:
        print("\n[noise_impact] Generating comparison study graphs...")

        _plot_study(
            eps,
            results["recurrent"]["clean"],
            results["noisy_recurrent"]["clean"],
            "RecurPPO brady_deg",
            "RecurPPO noise-trained",
            "Study A — Clean Environment",
            "No noise (σ = 0)",
            os.path.join(OUT_DIR, "study_clean_env.png"),
        )

        _plot_study(
            eps,
            results["recurrent"]["study_noisy"],
            results["noisy_recurrent"]["study_noisy"],
            "RecurPPO brady_deg",
            "RecurPPO noise-trained",
            "Study B — Noisy Environment",
            f"Fixed noise σ = {SIGMA_STUDY} (~20 dB SNR)",
            os.path.join(OUT_DIR, "study_noisy_env.png"),
        )
    elif "recurrent" in results or "noisy_recurrent" in results:
        missing = "noisy_recurrent" if "recurrent" in results else "recurrent"
        print(f"\n[noise_impact] Skipping study graphs — {missing} policy not found.")
        print("  Train it first, then re-run noise_impact.py.")

    print("\n" + "=" * 60)
    print("[noise_impact] Done. Outputs in results/noise/")
    print("=" * 60)


if __name__ == "__main__":
    main()
