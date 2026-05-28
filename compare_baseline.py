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
