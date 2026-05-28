# Baseline vs RecPPO Policy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `compare_baseline.py` that evaluates MLP deg-only (`policy_deg`) and RecPPO (`policy_brady_deg_recurrent`) on the same 80 brady+deg episodes, then outputs confusion matrices and a 4-panel comparison figure.

**Architecture:** Single self-contained script that imports helpers from `evaluation.py` (`plan_trials`, `severity_quartile_to_range`, `angle_bin_to_target`) and `utils.py` (`plot_confusion_matrix`, `compute_severity`, `get_angle_bin`, `get_severity_quartile`). One shared exo env instance is reset to the same patient state for each policy track. Four tracks per episode: healthy, no-exo, baseline, RecPPO. Seeds are deterministic so the 80 episodes are fully reproducible.

**Tech Stack:** Python 3.12, stable-baselines3 (PPO), sb3-contrib (RecurrentPPO), myosuite, numpy, matplotlib, scipy

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `compare_baseline.py` | Full evaluation + plotting script |
| Create | `tests/test_compare_baseline.py` | Unit tests for pure functions |
| Read (no change) | `evaluation.py` | Import `plan_trials`, `severity_quartile_to_range`, `angle_bin_to_target` |
| Read (no change) | `utils.py` | Import `plot_confusion_matrix`, `compute_severity`, `get_angle_bin`, `get_severity_quartile` |
| Read (no change) | `envs/elbow_env.py` | Import `CombinedExoOnlyWrapper` |

---

## Task 1: Skeleton, Constants, and Imports

**Files:**
- Create: `compare_baseline.py`

- [ ] **Step 1: Create the file with all imports and constants**

```python
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
```

- [ ] **Step 2: Verify the imports work (no environments loaded yet)**

```
python -c "
import sys; sys.argv = ['x']
from evaluation import plan_trials, severity_quartile_to_range, angle_bin_to_target
from utils import plot_confusion_matrix, compute_severity, get_angle_bin, get_severity_quartile
print('imports ok')
"
```

Expected: `imports ok` with no error.

- [ ] **Step 3: Commit skeleton**

```
git add compare_baseline.py
git commit -m "feat: add compare_baseline.py skeleton with imports and constants"
```

---

## Task 2: Patient Configuration and Track Helpers

**Files:**
- Modify: `compare_baseline.py` (add functions)
- Create: `tests/test_compare_baseline.py`

- [ ] **Step 1: Write failing unit tests for pure helper functions**

Create `tests/test_compare_baseline.py`:

```python
import numpy as np
import pytest

# We test only pure, env-free functions. Import after the module exists.

def test_build_matrix_fills_correct_cells():
    from compare_baseline import build_matrix

    severity_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    angle_edges    = np.linspace(0.0, 2.27, 5)  # 4 bins

    trials = [
        {"angle_bin": 0, "severity": 0.1, "test_corr": 0.8},
        {"angle_bin": 0, "severity": 0.1, "test_corr": 0.6},
        {"angle_bin": 1, "severity": 0.6, "test_corr": 0.5},
    ]

    matrix = build_matrix(trials, "test_corr", severity_edges, angle_edges)

    assert matrix.shape == (4, 4)
    # Cell (0,0): mean of 0.8 and 0.6
    assert abs(matrix[0, 0] - 0.7) < 1e-9
    # Cell (1,2): 0.5
    assert abs(matrix[1, 2] - 0.5) < 1e-9
    # Unfilled cells are NaN
    assert np.isnan(matrix[0, 1])


def test_build_matrix_empty_returns_all_nan():
    from compare_baseline import build_matrix

    severity_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    angle_edges    = np.linspace(0.0, 2.27, 5)
    matrix = build_matrix([], "corr", severity_edges, angle_edges)
    assert matrix.shape == (4, 4)
    assert np.all(np.isnan(matrix))


def test_compute_boost_pct():
    from compare_baseline import compute_boost_pct

    # 50% of the remaining gap from 0.4 floor to 1.0 ceiling
    assert abs(compute_boost_pct(acc=0.7, floor=0.4) - 50.0) < 1e-9
    # At floor — 0% boost
    assert compute_boost_pct(acc=0.4, floor=0.4) == 0.0
    # At ceiling — 100% boost
    assert abs(compute_boost_pct(acc=1.0, floor=0.4) - 100.0) < 1e-9
```

- [ ] **Step 2: Run tests — expect ImportError (functions not yet defined)**

```
python -m pytest tests/test_compare_baseline.py -v
```

Expected: `ImportError: cannot import name 'build_matrix' from 'compare_baseline'`

- [ ] **Step 3: Add `configure_patient`, `build_matrix`, `compute_boost_pct`, and track helpers to `compare_baseline.py`**

Append after the constants block:

```python
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
    angle_edges: np.ndarray,
) -> np.ndarray:
    """
    Build 4×4 Pearson-r matrix (angle bins × severity quartiles).
    Cells with no trials are NaN.
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
```

- [ ] **Step 4: Run the unit tests — expect PASS**

```
python -m pytest tests/test_compare_baseline.py -v
```

Expected:
```
test_compare_baseline.py::test_build_matrix_fills_correct_cells PASSED
test_compare_baseline.py::test_build_matrix_empty_returns_all_nan PASSED
test_compare_baseline.py::test_compute_boost_pct PASSED
```

- [ ] **Step 5: Commit**

```
git add compare_baseline.py tests/test_compare_baseline.py
git commit -m "feat: add patient config, track runners, build_matrix, compute_boost_pct"
```

---

## Task 3: Full Trial Runner

**Files:**
- Modify: `compare_baseline.py` (add `run_trial`)

- [ ] **Step 1: Add `run_trial` function**

Append to `compare_baseline.py`:

```python
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
    and applied identically to tracks 2–4.
    """
    # Sample patient state
    target_angle = angle_bin_to_target(angle_bin, angle_edges)
    fs_range, sl_range, mf_range = severity_quartile_to_range(sev_quartile)
    force_scale        = float(np.random.uniform(*fs_range))
    activation_slowdown = float(np.random.uniform(*sl_range))
    avg_mf_target      = float(np.random.uniform(*mf_range))

    n_muscles = base_env.n_muscles
    mf_vals   = np.random.uniform(
        max(avg_mf_target * 0.9, 0.0),
        min(avg_mf_target * 1.1, 1.0),
        size=n_muscles,
    )
    split_vals = np.random.uniform(0.0, 1.0, size=n_muscles)

    # Track 1: Healthy
    healthy_angles = run_healthy_track(healthy_env, healthy_policy, target_angle)

    # Compute actual avg MF for severity calculation
    actual_avg_mf = float(np.mean(mf_vals))
    severity = compute_severity(force_scale, activation_slowdown, actual_avg_mf)

    # Track 2: No-exo
    exo_env.reset()
    configure_patient(base_env, target_angle, force_scale,
                      activation_slowdown, mf_vals, split_vals)
    zero_action = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    no_exo_angles = []
    obs, _ = exo_env.reset()
    configure_patient(base_env, target_angle, force_scale,
                      activation_slowdown, mf_vals, split_vals)
    for _ in range(MAX_STEPS):
        obs, _, done, truncated, _ = exo_env.step(zero_action)
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

    # Pearson r calculations
    def _safe_r(a, b):
        n = min(len(a), len(b))
        if n < 2:
            return 0.0
        raw_r, _ = pearsonr(a[:n], b[:n])
        return float(raw_r) if not np.isnan(raw_r) else 0.0

    return {
        "angle_bin":        angle_bin,
        "sev_quartile":     sev_quartile,
        "target_angle":     target_angle,
        "force_scale":      force_scale,
        "activation_slowdown": activation_slowdown,
        "avg_mf":           actual_avg_mf,
        "severity":         severity,
        # correlations
        "baseline_corr":    _safe_r(baseline_result["angles"],  healthy_angles),
        "recurrent_corr":   _safe_r(recurrent_result["angles"], healthy_angles),
        "no_exo_corr":      _safe_r(no_exo_angles,             healthy_angles),
        # rewards
        "baseline_reward":  baseline_result["reward"],
        "recurrent_reward": recurrent_result["reward"],
        # goal info
        "baseline_goal":    baseline_result["goal_achieved"],
        "recurrent_goal":   recurrent_result["goal_achieved"],
    }
```

- [ ] **Step 2: Verify syntax**

```
python -c "import compare_baseline; print('run_trial defined:', hasattr(compare_baseline, 'run_trial'))"
```

Expected: `run_trial defined: True`

- [ ] **Step 3: Commit**

```
git add compare_baseline.py
git commit -m "feat: add run_trial with 4 tracks (healthy, no-exo, baseline, RecPPO)"
```

---

## Task 4: Plotting — 4-Panel Combined Figure

**Files:**
- Modify: `compare_baseline.py` (add `plot_comparison_metrics`)

- [ ] **Step 1: Add `plot_comparison_metrics`**

Append to `compare_baseline.py`:

```python
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison_metrics(trials: list, no_exo_mean: float, out_dir: str) -> None:
    """
    4-panel combined figure:
      Panel 1 (top-left)  : Reward per episode
      Panel 2 (top-right) : Mean Pearson r bar chart (baseline vs RecPPO vs no-exo)
      Panel 3 (bottom-left): Pearson r per episode — sequential with gap shading
      Panel 4 (bottom-right): Pearson r per episode — sorted by severity
    """
    n = len(trials)
    eps = np.arange(1, n + 1)

    baseline_rewards  = [t["baseline_reward"]  for t in trials]
    recurrent_rewards = [t["recurrent_reward"] for t in trials]
    baseline_r        = [t["baseline_corr"]    for t in trials]
    recurrent_r       = [t["recurrent_corr"]   for t in trials]
    severities        = [t["severity"]          for t in trials]

    mean_base = float(np.mean(baseline_r))
    mean_rec  = float(np.mean(recurrent_r))
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
    labels = ["MLP deg-only\n(baseline)", "RecPPO\nbrady+deg", "No-Exo\nfloor"]
    values = [mean_base, mean_rec, no_exo_mean]
    colors = ["steelblue", "coral", "gray"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
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
    br = np.array(baseline_r)
    rr = np.array(recurrent_r)
    ax.plot(eps, br, color="steelblue", linewidth=1.2, label="MLP deg-only")
    ax.plot(eps, rr, color="coral",     linewidth=1.2, label="RecPPO")
    ax.fill_between(eps, br, rr,
                    where=(rr >= br), alpha=0.15, color="coral",
                    label="RecPPO advantage")
    ax.fill_between(eps, br, rr,
                    where=(rr < br), alpha=0.15, color="steelblue",
                    label="Baseline advantage")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="Healthy")
    ax.axhline(no_exo_mean, color="gray", linestyle=":", linewidth=0.8,
               label=f"No-exo floor ({no_exo_mean:.3f})")
    ax.set_xlabel("Episode (sequential)")
    ax.set_ylabel("Pearson r (vs Healthy)")
    ax.set_title("Pearson r Per Episode — Sequential")
    ax.legend(fontsize=7)

    # --- Panel 4: Pearson r sorted by severity ---
    sort_idx = np.argsort(severities)
    sorted_sev = np.array(severities)[sort_idx]
    sorted_br  = br[sort_idx]
    sorted_rr  = rr[sort_idx]
    ax = axes[1, 1]
    ax.plot(range(n), sorted_br, color="steelblue", linewidth=1.2, label="MLP deg-only")
    ax.plot(range(n), sorted_rr, color="coral",     linewidth=1.2, label="RecPPO")
    ax.fill_between(range(n), sorted_br, sorted_rr,
                    where=(sorted_rr >= sorted_br), alpha=0.15, color="coral")
    ax.fill_between(range(n), sorted_br, sorted_rr,
                    where=(sorted_rr < sorted_br), alpha=0.15, color="steelblue")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.axhline(no_exo_mean, color="gray", linestyle=":", linewidth=0.8)
    # Mark severity quartile boundaries
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
    print(f"  Saved → {out_path}")
```

- [ ] **Step 2: Verify syntax**

```
python -c "import compare_baseline; print('plot_comparison_metrics defined:', hasattr(compare_baseline, 'plot_comparison_metrics'))"
```

Expected: `plot_comparison_metrics defined: True`

- [ ] **Step 3: Commit**

```
git add compare_baseline.py
git commit -m "feat: add 4-panel comparison_metrics plot"
```

---

## Task 5: Severity-Sorted Standalone Plot

**Files:**
- Modify: `compare_baseline.py` (add `plot_pearsonr_by_severity`)

- [ ] **Step 1: Add `plot_pearsonr_by_severity`**

Append to `compare_baseline.py`:

```python
def plot_pearsonr_by_severity(trials: list, no_exo_mean: float, out_dir: str) -> None:
    """
    Standalone larger version of Panel 4 — Pearson r sorted by severity.
    Easier to include in a report than the 4-panel figure.
    """
    severities  = np.array([t["severity"]       for t in trials])
    baseline_r  = np.array([t["baseline_corr"]  for t in trials])
    recurrent_r = np.array([t["recurrent_corr"] for t in trials])

    sort_idx    = np.argsort(severities)
    sorted_sev  = severities[sort_idx]
    sorted_br   = baseline_r[sort_idx]
    sorted_rr   = recurrent_r[sort_idx]
    n           = len(trials)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(n), sorted_br, color="steelblue", linewidth=1.5,
            label="MLP deg-only (baseline)")
    ax.plot(range(n), sorted_rr, color="coral",     linewidth=1.5,
            label="RecPPO brady+deg")
    ax.fill_between(range(n), sorted_br, sorted_rr,
                    where=(sorted_rr >= sorted_br), alpha=0.18, color="coral",
                    label="RecPPO advantage")
    ax.fill_between(range(n), sorted_br, sorted_rr,
                    where=(sorted_rr < sorted_br), alpha=0.18, color="steelblue",
                    label="Baseline advantage")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0,
               label="Healthy (Pearson r = 1.0)")
    ax.axhline(no_exo_mean, color="gray", linestyle=":", linewidth=1.0,
               label=f"No-exo floor ({no_exo_mean:.3f})")

    # Severity quartile boundary markers
    q_edges = np.percentile(sorted_sev, [25, 50, 75])
    for qi, q in enumerate(q_edges):
        idx = int(np.searchsorted(sorted_sev, q))
        ax.axvline(idx, color="dimgray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.text(idx + 0.3, 0.02, f"Q{qi+2}", fontsize=8, color="dimgray")

    ax.set_xlabel("Episodes sorted by patient severity (mild → severe)", fontsize=11)
    ax.set_ylabel("Pearson r (vs Healthy)", fontsize=11)
    ax.set_title("Policy Performance vs Patient Severity — Baseline vs RecPPO",
                 fontsize=12)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "pearsonr_by_severity.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")
```

- [ ] **Step 2: Verify syntax**

```
python -c "import compare_baseline; print('plot_pearsonr_by_severity defined:', hasattr(compare_baseline, 'plot_pearsonr_by_severity'))"
```

Expected: `plot_pearsonr_by_severity defined: True`

- [ ] **Step 3: Commit**

```
git add compare_baseline.py
git commit -m "feat: add standalone pearsonr_by_severity plot"
```

---

## Task 6: Main Function and End-to-End Run

**Files:**
- Modify: `compare_baseline.py` (add `main`)

- [ ] **Step 1: Add `main()`**

Append to `compare_baseline.py`:

```python
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not _HAS_RECURRENT_PPO:
        raise ImportError("sb3_contrib not installed — run: pip install sb3-contrib")

    for path, name in [
        (HEALTHY_PATH,   "healthy_policy.zip"),
        (BASELINE_PATH,  "policy_deg.zip"),
        (RECURRENT_PATH, "policy_brady_deg_recurrent.zip"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Policy not found: {path}")

    print("=" * 60)
    print("reGainX — Baseline vs RecPPO Comparison")
    print(f"  Episodes  : {N_EPISODES}  (5 per cell, 4×4 grid)")
    print(f"  Baseline  : policy_deg (MLP, deg-only trained)")
    print(f"  RecPPO    : policy_brady_deg_recurrent")
    print(f"  Env       : brady+deg (bradykinesia=True, smart_reset=True)")
    print(f"  Output    : {OUT_DIR}/")
    print("=" * 60)

    # Load policies
    healthy_policy   = PPO.load(HEALTHY_PATH)
    baseline_policy  = PPO.load(BASELINE_PATH)
    recurrent_policy = RecurrentPPO.load(RECURRENT_PATH)

    # Build environments
    healthy_env = gym.make("myoElbowPose1D6MRandom-v0")
    base_raw    = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    exo_env     = CombinedExoOnlyWrapper(
        base_raw,
        frozen_policy_path=HEALTHY_PATH,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )
    base_env = exo_env  # CombinedExoOnlyWrapper IS the base_env (no LSTM wrapper)

    # Angle edges from env range
    tmp = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    low  = float(tmp.unwrapped.target_jnt_range[0, 0])
    high = float(tmp.unwrapped.target_jnt_range[0, 1])
    tmp.close()
    angle_edges = np.linspace(low, high, ANGLE_BINS + 1)

    # Episode plan
    trial_plan = plan_trials(N_EPISODES)
    print(f"\nRunning {N_EPISODES} trials across {ANGLE_BINS}×{SEVERITY_BINS} matrix cells")

    all_trials = []
    for i, (angle_bin, sev_quartile) in enumerate(trial_plan):
        print(f"  Trial {i+1:3d}/{N_EPISODES}  cell=({angle_bin},{sev_quartile})", end="  ")
        t = run_trial(
            exo_env, base_env, healthy_env,
            baseline_policy, recurrent_policy, healthy_policy,
            angle_bin, sev_quartile, angle_edges,
        )
        all_trials.append(t)
        print(f"base_r={t['baseline_corr']:.3f}  rec_r={t['recurrent_corr']:.3f}  "
              f"sev={t['severity']:.2f}")

    healthy_env.close()
    exo_env.close()

    # Severity edges from actual observed severities
    all_sev = [t["severity"] for t in all_trials]
    severity_edges = np.percentile(all_sev, [0, 25, 50, 75, 100])

    no_exo_mean = float(np.mean([t["no_exo_corr"] for t in all_trials]))
    mean_base   = float(np.mean([t["baseline_corr"]  for t in all_trials]))
    mean_rec    = float(np.mean([t["recurrent_corr"] for t in all_trials]))

    # Confusion matrices
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\nGenerating plots → {OUT_DIR}/")

    matrix_base   = build_matrix(all_trials, "baseline_corr",  severity_edges, angle_edges)
    matrix_rec    = build_matrix(all_trials, "recurrent_corr", severity_edges, angle_edges)
    matrix_no_exo = build_matrix(all_trials, "no_exo_corr",    severity_edges, angle_edges)

    plot_confusion_matrix(
        matrix_base, ANGLE_LABELS, SEVERITY_LABELS,
        "Movement Accuracy (MLP deg-only on brady+deg env)",
        os.path.join(OUT_DIR, "confusion_matrix_baseline.png"), pct=True,
    )
    plot_confusion_matrix(
        matrix_rec, ANGLE_LABELS, SEVERITY_LABELS,
        "Movement Accuracy (RecPPO brady+deg)",
        os.path.join(OUT_DIR, "confusion_matrix_recppo.png"), pct=True,
    )
    plot_confusion_matrix(
        matrix_no_exo, ANGLE_LABELS, SEVERITY_LABELS,
        "Movement Accuracy (No-Exo Floor)",
        os.path.join(OUT_DIR, "confusion_matrix_no_exo.png"), pct=True,
    )

    plot_comparison_metrics(all_trials, no_exo_mean, OUT_DIR)
    plot_pearsonr_by_severity(all_trials, no_exo_mean, OUT_DIR)

    # Summary
    boost_base = compute_boost_pct(mean_base, no_exo_mean)
    boost_rec  = compute_boost_pct(mean_rec,  no_exo_mean)
    rec_adv    = mean_rec - mean_base

    print(f"\n{'='*60}")
    print("Baseline Comparison Summary")
    print(f"  Episodes         : {N_EPISODES}")
    print(f"  Baseline policy  : policy_deg (MLP, deg-only trained)")
    print(f"  RecPPO policy    : policy_brady_deg_recurrent")
    print()
    print(f"  Mean Pearson r  (Baseline) : {mean_base:.4f}")
    print(f"  Mean Pearson r  (RecPPO)   : {mean_rec:.4f}")
    print(f"  Mean Pearson r  (No-exo)   : {no_exo_mean:.4f}")
    print(f"  Boost Baseline             : +{boost_base:.1f}% toward healthy")
    print(f"  Boost RecPPO               : +{boost_rec:.1f}% toward healthy")
    print(f"  RecPPO advantage           : {rec_adv:+.4f} Pearson r over baseline")
    print(f"{'='*60}")
    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"  confusion_matrix_baseline.png")
    print(f"  confusion_matrix_recppo.png")
    print(f"  confusion_matrix_no_exo.png")
    print(f"  comparison_metrics.png")
    print(f"  pearsonr_by_severity.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify full syntax + import**

```
python -c "import compare_baseline; print('main defined:', hasattr(compare_baseline, 'main'))"
```

Expected: `main defined: True`

- [ ] **Step 3: Run the unit tests once more to confirm nothing broke**

```
python -m pytest tests/test_compare_baseline.py -v
```

Expected: 3 tests PASSED.

- [ ] **Step 4: Run the script end-to-end**

```
python compare_baseline.py
```

Expected terminal output pattern:
```
============================================================
reGainX — Baseline vs RecPPO Comparison
  Episodes  : 80  (5 per cell, 4×4 grid)
...
Running 80 trials across 4×4 matrix cells
  Trial   1/80  cell=(0,0)  base_r=0.xxx  rec_r=0.xxx  sev=0.xx
  ...
  Trial  80/80  cell=(3,3)  base_r=0.xxx  rec_r=0.xxx  sev=0.xx

Generating plots → results/baseline_comparison/
  Saved → results/baseline_comparison/comparison_metrics.png
  Saved → results/baseline_comparison/pearsonr_by_severity.png

============================================================
Baseline Comparison Summary
...
============================================================
```

Expected files created (verify these exist):
```
python -c "
import os
files = [
    'results/baseline_comparison/confusion_matrix_baseline.png',
    'results/baseline_comparison/confusion_matrix_recppo.png',
    'results/baseline_comparison/confusion_matrix_no_exo.png',
    'results/baseline_comparison/comparison_metrics.png',
    'results/baseline_comparison/pearsonr_by_severity.png',
]
for f in files:
    status = 'OK' if os.path.exists(f) else 'MISSING'
    print(f'{status}: {f}')
"
```

Expected: All 5 files show `OK`.

- [ ] **Step 5: Final commit**

```
git add compare_baseline.py tests/test_compare_baseline.py
git commit -m "feat: complete compare_baseline.py — MLP deg-only vs RecPPO on brady+deg env"
```

---

## Self-Review

**Spec coverage:**
- [x] Script named `compare_baseline.py` → ✓ Task 1
- [x] Output to `results/baseline_comparison/` → ✓ Task 6 (OUT_DIR constant)
- [x] Baseline = `policy_deg.zip` (MLP deg-only) → ✓ BASELINE_PATH hardcoded
- [x] RecPPO = `policy_brady_deg_recurrent.zip` → ✓ RECURRENT_PATH hardcoded
- [x] Environment: bradykinesia=True + smart_reset=True → ✓ Task 6 env construction
- [x] 80 episodes, 5 per cell (4×4 grid) → ✓ `plan_trials(80)` in Task 6
- [x] Same patient state for both policies → ✓ Task 3 `run_trial` uses same mf_vals/split_vals
- [x] Pearson r vs healthy for both policies → ✓ Task 3 `_safe_r` calls
- [x] Confusion matrices (baseline, recppo, no-exo) → ✓ Task 6 `build_matrix` + `plot_confusion_matrix`
- [x] 4-panel combined figure → ✓ Task 4 panels 1-4
- [x] Pearson r by severity (separate PNG) → ✓ Task 5
- [x] No changes to existing files → ✓ only imports from them

**Placeholder scan:** None found. All steps contain complete code.

**Type consistency:**
- `build_matrix(trials, corr_key, severity_edges, angle_edges)` — used consistently in Task 3 (unit test) and Task 6 (main)
- `compute_boost_pct(acc, floor)` — tested in Task 2, used in Task 4 and Task 6
- `run_exo_track(...)` returns `{"angles", "reward", "goal_achieved", "goal_time"}` — consumed correctly in `run_trial`
- `run_trial(...)` returns keys `["baseline_corr", "recurrent_corr", "no_exo_corr", "baseline_reward", "recurrent_reward", "severity", "angle_bin", ...]` — all consumed in Task 4, 5, 6
