"""
Cross-policy accuracy comparison: Pearson r vs healthy baseline.

100 shared-seed episodes. Three tracks per episode:
  1. Healthy  — healthy policy on myoElbowPose1D6MRandom-v0 (no impairments)
  2. No-exo   — zero action on myoFatiElbowPose1D6MExoRandom-v0 (impaired)
  3. Each exo policy on the same impaired env

Accuracy = mean Pearson r (exo trajectory vs healthy trajectory) over episodes.
Boost    = (accuracy - no_exo_accuracy) / (1.0 - no_exo_accuracy) * 100 %

Outputs:
  results/algo_compare/accuracy_bar.png
  results/algo_compare/accuracy_per_episode.png

Usage:
  python algo_compare.py --healthy policies/healthy_policy.zip [options]
"""

import argparse
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
from envs.temporal_buffer import TemporalStackWrapper

# r_elbow_flex range confirmed from MyoSuite installed source
JOINT_LOW  = 0.0
JOINT_HIGH = 2.27
MAX_STEPS  = 200  # gym truncates at 100; this is just a guard
OUT_DIR    = "results/algo_compare"


# ---------------------------------------------------------------------------
# Environment builders
# ---------------------------------------------------------------------------

def _make_healthy_env():
    return gym.make("myoElbowPose1D6MRandom-v0")


def _make_exo_env(healthy_path: str, lstm: bool = False, extra_obs: bool = False):
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=healthy_path,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=extra_obs,
    )
    if lstm:
        env = TemporalStackWrapper(env, window=20)
    return env


def _load_policy(path: str, is_recurrent: bool = False):
    if is_recurrent:
        if not _HAS_RECURRENT_PPO:
            raise ImportError("sb3_contrib not installed — run: pip install sb3-contrib")
        return RecurrentPPO.load(path)
    return PPO.load(path)


# ---------------------------------------------------------------------------
# Patient-state seeding (shared across all envs for a given seed)
# ---------------------------------------------------------------------------

def _sample_patient_state(rng: np.random.RandomState, n_muscles: int,
                           force_scale_range: tuple, activation_slowdown_range: tuple):
    """Draw patient parameters from rng in a fixed order."""
    target_angle      = rng.uniform(JOINT_LOW, JOINT_HIGH)
    MF                = rng.uniform(0.7, 1.0, size=n_muscles)
    remaining         = 1.0 - MF
    split             = rng.uniform(0.0, 1.0, size=n_muscles)
    force_scale       = rng.uniform(*force_scale_range)
    activation_slow   = rng.uniform(*activation_slowdown_range)
    return target_angle, MF, remaining, split, force_scale, activation_slow


def _configure_exo_env(env, base_env, target_angle, MF, remaining, split,
                        force_scale, activation_slow):
    """Reset env and install seeded patient state. Returns initial obs."""
    is_lstm = isinstance(env, TemporalStackWrapper)
    env.reset()  # obs discarded — rebuilt from seeded patient state below
    base_env.base_env.unwrapped.target_jnt_value = [target_angle]
    base_env.base_env.unwrapped.target_type = "fixed"
    base_env.base_env.unwrapped.update_target(restore_sim=True)
    base_env.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split
    base_env.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split)
    base_env.base_env.unwrapped.muscle_fatigue.MF[:] = MF
    base_env.force_scale       = force_scale
    base_env.activation_slowdown = activation_slow
    base_env._apply_brady()
    base_env.base_env.unwrapped.sim.data.qpos[:] = 0.0
    base_env.base_env.unwrapped.sim.data.qvel[:] = 0.0
    base_env.base_env.unwrapped.sim.forward()
    raw      = base_env._current_raw_obs()
    obs_flat = base_env._build_obs(raw)
    if is_lstm:
        env._buffer.clear()
        for _ in range(env.window):
            env._buffer.append(obs_flat.copy())
        return env._stack()
    return obs_flat


def _pearsonr_safe(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    raw_r, _ = pearsonr(a[:n], b[:n])
    return float(raw_r) if not np.isnan(raw_r) else 0.0


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def _run_healthy_track(healthy_env, healthy_policy: PPO, target_angle: float):
    """Run healthy policy to target. Returns list of qpos[0] values."""
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


def _run_no_exo_track(exo_env, base_env, target_angle, MF, remaining, split,
                       force_scale, activation_slow):
    """Run zero-action on impaired env. Returns list of qpos[0] values."""
    obs = _configure_exo_env(exo_env, base_env, target_angle, MF, remaining,
                              split, force_scale, activation_slow)
    zero = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    angles = []
    # zero action every step; obs is updated by wrapper but not used to steer (no policy)
    for _ in range(MAX_STEPS):
        obs, _, done, truncated, _ = exo_env.step(zero)
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles


def _run_exo_track(exo_env, base_env, exo_policy, target_angle, MF, remaining,
                    split, force_scale, activation_slow, is_recurrent: bool):
    """Run exo policy on impaired env. Returns list of qpos[0] values."""
    obs = _configure_exo_env(exo_env, base_env, target_angle, MF, remaining,
                              split, force_scale, activation_slow)
    angles = []
    lstm_states   = None
    episode_start = np.ones((1,), dtype=bool)
    for _ in range(MAX_STEPS):
        if is_recurrent:
            action, lstm_states = exo_policy.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True)
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = exo_policy.predict(obs, deterministic=True)
        obs, _, done, truncated, _ = exo_env.step(action)
        angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break
    return angles


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_bar(policy_labels, acc_exo, acc_no_exo, boost_pct, out_path):
    n = len(policy_labels)
    x = np.arange(n)
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(9, n * 1.8), 6))

    ax.bar(x - w / 2, acc_no_exo, w, label="Impaired, no exo", color="gray", alpha=0.75)
    for i in range(n):
        clean_lbl = policy_labels[i].replace("\n", " ")
        bar = ax.bar(x[i] + w / 2, acc_exo[i], w, color=f"C{i}", label=clean_lbl)
        ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                min(acc_exo[i] + 0.015, 1.12),
                f"{boost_pct[i]:+.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Healthy (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (vs Healthy)")
    ax.set_ylim(0, 1.18)
    ax.set_title("Algorithm Accuracy Comparison — Pearson r vs Healthy Baseline")

    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [algo_compare] Saved → {out_path}")


def _plot_line(policy_labels, per_episode_r, no_exo_mean, n_episodes, out_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    eps = range(1, n_episodes + 1)
    for i, (lbl, r_series) in enumerate(zip(policy_labels, per_episode_r)):
        ax.plot(eps, r_series, label=lbl, color=f"C{i}", linewidth=1.3)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Healthy")
    ax.axhline(no_exo_mean, color="gray", linestyle=":", linewidth=1.2,
               label=f"No-exo floor ({no_exo_mean:.3f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Pearson r (vs Healthy)")
    ax.set_title("Per-Episode Accuracy — All Exo Policies vs Healthy Baseline")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  [algo_compare] Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-policy accuracy comparison")
    parser.add_argument("--healthy",        required=True)
    parser.add_argument("--brady",          default="")
    parser.add_argument("--deg",            default="")
    parser.add_argument("--lstm",           default="")
    parser.add_argument("--deg-lstm",       default="")
    parser.add_argument("--extraobs",       default="")
    parser.add_argument("--recurrent",      default="")
    parser.add_argument("--deg-recurrent",  default="")
    parser.add_argument("--episodes",       type=int, default=100)
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    # (path, lstm, extra_obs, is_recurrent, display_label)
    CONFIGS = [
        (args.brady,        False, False, False, "MLP\nbrady_deg"),
        (args.deg,          False, False, False, "MLP\ndeg"),
        (args.lstm,         True,  False, False, "CNN-LSTM\nbrady_deg"),
        (args.deg_lstm,     True,  False, False, "CNN-LSTM\ndeg"),
        (args.extraobs,     True,  True,  False, "CNN-LSTM\nextraobs"),
        (args.recurrent,    False, False, True,  "RecPPO\nbrady_deg"),
        (args.deg_recurrent,False, False, True,  "RecPPO\ndeg"),
    ]
    active = [(p, lstm, eo, rec, lbl) for (p, lstm, eo, rec, lbl) in CONFIGS
              if p and os.path.exists(p)]

    if not active:
        print("[algo_compare] No policy files found — skipping.")
        return

    print("=" * 60)
    print("[algo_compare] Cross-policy accuracy comparison")
    print(f"  Episodes : {args.episodes}")
    print(f"  Seed     : {args.seed}")
    print(f"  Policies : {len(active)}")
    print("=" * 60)

    seeds = np.random.default_rng(args.seed).integers(0, 2**31, size=args.episodes)
    healthy_policy = PPO.load(args.healthy)
    healthy_env    = _make_healthy_env()
    base_exo_env   = _make_exo_env(args.healthy, lstm=False, extra_obs=False)
    base_env_inner = base_exo_env  # CombinedExoOnlyWrapper (non-LSTM)

    n_muscles             = base_env_inner.n_muscles
    force_scale_range     = base_env_inner.force_scale_range
    activation_slow_range = base_env_inner.activation_slowdown_range

    # -- Phase 1: healthy + no-exo tracks (run once, shared across all policies) --
    print("\n[algo_compare] Phase 1: healthy + no-exo tracks...")
    all_healthy_angles = []
    r_no_exo_per_ep    = []

    for ep, seed in enumerate(seeds):
        rng = np.random.RandomState(int(seed))
        target_angle, MF, remaining, split, force_scale, activation_slow = \
            _sample_patient_state(rng, n_muscles, force_scale_range, activation_slow_range)

        healthy_angles  = _run_healthy_track(healthy_env, healthy_policy, target_angle)
        no_exo_angles   = _run_no_exo_track(base_exo_env, base_env_inner,
                                             target_angle, MF, remaining, split,
                                             force_scale, activation_slow)
        r_no_exo        = _pearsonr_safe(no_exo_angles, healthy_angles)
        all_healthy_angles.append(healthy_angles)
        r_no_exo_per_ep.append(r_no_exo)
        print(f"  ep {ep+1:3d}/{args.episodes}  no_exo_r={r_no_exo:.3f}", end="\r")

    print()
    no_exo_mean = float(np.mean(r_no_exo_per_ep))
    print(f"  No-exo mean Pearson r: {no_exo_mean:.4f}")
    base_exo_env.close()

    # -- Phase 2: per-policy exo tracks --
    policy_labels   = []
    all_acc_exo     = []
    all_per_ep_r    = []

    for path, lstm, extra_obs, is_recurrent, label in active:
        print(f"\n[algo_compare] Running: {label.replace(chr(10), ' ')}")
        exo_env  = _make_exo_env(args.healthy, lstm=lstm, extra_obs=extra_obs)
        base_env = exo_env.env if isinstance(exo_env, TemporalStackWrapper) else exo_env
        policy   = _load_policy(path, is_recurrent=is_recurrent)

        r_exo_per_ep = []
        for ep, seed in enumerate(seeds):
            rng = np.random.RandomState(int(seed))
            target_angle, MF, remaining, split, force_scale, activation_slow = \
                _sample_patient_state(rng, n_muscles, force_scale_range, activation_slow_range)

            exo_angles = _run_exo_track(exo_env, base_env, policy, target_angle,
                                         MF, remaining, split, force_scale,
                                         activation_slow, is_recurrent)
            r_exo = _pearsonr_safe(exo_angles, all_healthy_angles[ep])
            r_exo_per_ep.append(r_exo)
            print(f"  ep {ep+1:3d}/{args.episodes}  exo_r={r_exo:.3f}", end="\r")

        print()
        exo_env.close()
        acc = float(np.mean(r_exo_per_ep))
        print(f"  Accuracy: {acc:.4f}  (no-exo floor: {no_exo_mean:.4f})")
        policy_labels.append(label)
        all_acc_exo.append(acc)
        all_per_ep_r.append(r_exo_per_ep)

    healthy_env.close()

    # -- Compute boost --
    gap        = max(1.0 - no_exo_mean, 1e-9)
    boost_pct  = [(acc - no_exo_mean) / gap * 100 for acc in all_acc_exo]
    acc_no_exo = [no_exo_mean] * len(policy_labels)

    # -- Plot --
    _plot_bar(policy_labels, all_acc_exo, acc_no_exo, boost_pct,
              os.path.join(OUT_DIR, "accuracy_bar.png"))
    _plot_line(policy_labels, all_per_ep_r, no_exo_mean, args.episodes,
               os.path.join(OUT_DIR, "accuracy_per_episode.png"))

    print("\n" + "=" * 60)
    print("[algo_compare] Summary")
    for lbl, acc, boost in zip(policy_labels, all_acc_exo, boost_pct):
        print(f"  {lbl.replace(chr(10), ' '):40s}  r={acc:.4f}  boost={boost:+.1f}%")
    print(f"  {'No-exo floor':40s}  r={no_exo_mean:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
