# Algorithm Accuracy Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a cross-policy accuracy comparison to `pipeline.py` that runs 100 shared-seed episodes, computes Pearson r vs the healthy baseline for every exo policy, and outputs a grouped bar chart and per-episode line plot in `results/algo_compare/`. Also fix existing confusion matrices in `evaluation.py` and `compare.py` to show a no-exo baseline matrix alongside the with-exo matrix, both expressed as %.

**Architecture:** New standalone `algo_compare.py` runs 3 tracks per episode (healthy / no-exo / each exo policy) using shared `np.random.RandomState(seed)` sequences so all policies face identical patient states. `utils.py` gains a `pct` flag on `plot_confusion_matrix`. `evaluation.py` and `compare.py` each add a no-exo track and call the updated plotter with `pct=True`. `pipeline.py` adds Stage 5.

**Tech Stack:** Python, NumPy, SciPy (`pearsonr`), Matplotlib, Stable-Baselines3, sb3-contrib (`RecurrentPPO`), MyoSuite (`myoFatiElbowPose1D6MExoRandom-v0`, `myoElbowPose1D6MRandom-v0`)

**MyoSuite API facts (confirmed from installed source):**
- `r_elbow_flex` range: [0.0, 2.27] rad
- `max_episode_steps = 100` (gym truncates at step 100); use `MAX_STEPS = 200` as loop guard
- `target_jnt_value = np.array([float])` — 1-element array, not scalar
- Target-lock sequence: `env.unwrapped.target_jnt_value = np.array([angle])` → `env.unwrapped.target_type = "fixed"` → `env.unwrapped.update_target(restore_sim=True)`
- `rwd_dict["solved"]` = `norm(target_jnt_value - qpos) < 0.175`
- `qpos[0]` = `r_elbow_flex` angle

---

## Task 1: Update `plot_confusion_matrix` in `utils.py` to support `pct` mode

**Files:**
- Modify: `utils.py`
- Test: `tests/test_algo_compare.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_algo_compare.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
from utils import plot_confusion_matrix

def test_plot_confusion_matrix_pct_labels(tmp_path):
    matrix = np.array([[0.8, 0.6], [0.4, np.nan]])
    angle_labels = ["0.5-1.0", "1.0-1.5"]
    sev_labels   = ["Q1 mild", "Q2"]
    out = str(tmp_path / "cm.png")
    # should not raise; file should be created
    plot_confusion_matrix(matrix, angle_labels, sev_labels, "Test", out, pct=True)
    assert os.path.exists(out)

def test_plot_confusion_matrix_default_no_pct(tmp_path):
    matrix = np.array([[0.8, 0.6], [0.4, 0.2]])
    out = str(tmp_path / "cm.png")
    plot_confusion_matrix(matrix, ["a", "b"], ["c", "d"], "T", out)
    assert os.path.exists(out)
```

- [ ] **Step 2: Run test to verify it fails**

```
cd C:\Users\rohan\Downloads\reGainX_git
conda run -n exo_s python -m pytest tests/test_algo_compare.py::test_plot_confusion_matrix_pct_labels -v
```

Expected: `FAILED` — `TypeError: plot_confusion_matrix() got an unexpected keyword argument 'pct'`

- [ ] **Step 3: Update `plot_confusion_matrix` in `utils.py`**

Replace the entire `plot_confusion_matrix` function (lines 71–108) with:

```python
def plot_confusion_matrix(
    matrix: np.ndarray,
    angle_labels: list,
    severity_labels: list,
    title: str,
    save_path: str,
    pct: bool = False,
) -> None:
    """
    Blues confusion matrix: rows=angle bins, cols=severity quartiles.
    pct=True: multiply values by 100, label cells as %, vmax=100.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    display = np.nan_to_num(matrix, nan=0.0)
    if pct:
        display = display * 100.0
        vmax = 100.0
        cbar_label = "% trajectory match vs healthy"
    else:
        vmax = 1.0
        cbar_label = "Pearson r (higher = closer to healthy)"

    im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(severity_labels)))
    ax.set_xticklabels(severity_labels)
    ax.set_yticks(range(len(angle_labels)))
    ax.set_yticklabels(angle_labels)
    ax.set_xlabel("Severity Quartile")
    ax.set_ylabel("Target Angle (rad)")
    ax.set_title(title)

    for i in range(len(angle_labels)):
        for j in range(len(severity_labels)):
            val = matrix[i, j]
            if np.isnan(val):
                text = "N/A"
            elif pct:
                text = f"{val * 100:.0f}%"
            else:
                text = f"{val:.2f}"
            threshold = 60.0 if pct else 0.6
            color = "white" if display[i, j] > threshold else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

    plt.colorbar(im, ax=ax, label=cbar_label)
    plt.tight_layout()
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```
conda run -n exo_s python -m pytest tests/test_algo_compare.py -v
```

Expected: both tests `PASSED`

- [ ] **Step 5: Commit**

```
git add utils.py tests/test_algo_compare.py
git commit -m "feat: add pct mode to plot_confusion_matrix"
```

---

## Task 2: Add no-exo confusion matrix to `evaluation.py`

**Files:**
- Modify: `evaluation.py`

`evaluation.py` already collects `no_exo_angles` and `healthy_angles` per trial. This task:
1. Adds `no_exo_correlation` computation inside `run_eval_trial`
2. Builds a second confusion matrix from it in `main()`
3. Saves `confusion_matrix_no_exo.png` and renames the existing output to `confusion_matrix_with_exo.png`
4. Passes `pct=True` to both matrix plots

- [ ] **Step 1: Add `no_exo_correlation` to `run_eval_trial` return value**

In `evaluation.py`, inside `run_eval_trial`, the existing block at the bottom computes `corr` (exo vs healthy). After that block, add:

```python
    # no-exo vs healthy
    min_len_no = min(len(healthy_angles), len(no_exo_angles))
    no_exo_corr = 0.0
    if min_len_no > 1:
        no_exo_corr, _ = pearsonr(healthy_angles[:min_len_no], no_exo_angles[:min_len_no])
```

Then add `"no_exo_correlation": no_exo_corr` to the return dict (alongside the existing `"correlation": corr`). The full updated return statement (lines 219–233):

```python
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
```

- [ ] **Step 2: Add no-exo matrix tracking in `main()`**

In `evaluation.py`'s `main()`, after the existing matrix variable declarations (around line 406–410), add two more:

```python
    matrix_no_exo   = np.full((ANGLE_BINS, SEVERITY_BINS), np.nan)
    mat_no_exo_counts = np.zeros((ANGLE_BINS, SEVERITY_BINS), dtype=int)
    mat_no_exo_sums   = np.zeros((ANGLE_BINS, SEVERITY_BINS))
```

- [ ] **Step 3: Accumulate no-exo correlation in the trial loop**

Inside the trial loop in `main()`, after `mat_sums[angle_bin, sev_quartile] += trial["correlation"]` and the associated count/matrix update block, add:

```python
        mat_no_exo_sums[angle_bin, sev_quartile]   += trial["no_exo_correlation"]
        mat_no_exo_counts[angle_bin, sev_quartile] += 1
        if mat_no_exo_counts[angle_bin, sev_quartile] >= 1:
            matrix_no_exo[angle_bin, sev_quartile] = (
                mat_no_exo_sums[angle_bin, sev_quartile]
                / mat_no_exo_counts[angle_bin, sev_quartile]
            )
```

- [ ] **Step 4: Mark no-exo matrix NaN cells and save both matrices**

After the trial loop ends, where `matrix[mat_counts < 1] = np.nan` is set, add:

```python
    matrix_no_exo[mat_no_exo_counts < 1] = np.nan
```

Then find the existing `plot_confusion_matrix` call (around line 446) and replace it plus the surrounding calls with:

```python
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
```

- [ ] **Step 5: Smoke-test that the files are produced (dry-run with mock)**

```
conda run -n exo_s python -c "
import numpy as np
from utils import plot_confusion_matrix
m = np.random.uniform(0, 1, (4, 4))
plot_confusion_matrix(m, ['a','b','c','d'], ['Q1','Q2','Q3','Q4'], 'Test', '/tmp/cm_test.png', pct=True)
print('OK')
"
```

Expected: `OK`

- [ ] **Step 6: Commit**

```
git add evaluation.py
git commit -m "feat: add no-exo confusion matrix to evaluation.py, pct display"
```

---

## Task 3: Add no-exo track and confusion matrix to `compare.py`

**Files:**
- Modify: `compare.py`

`compare.py`'s `run_trial` currently runs two tracks (healthy + exo). This task adds a third zero-action track, returns `no_exo_correlation`, builds a no-exo matrix in `run_comparison`, and switches all matrix calls to `pct=True`.

- [ ] **Step 1: Add no-exo track at the end of `run_trial`**

In `compare.py`, inside `run_trial`, after the exo track loop ends and before the final `pearsonr` computation (around line 195), add:

```python
    # --- Track: no-exo (zero action, same patient state) ---
    exo_env.reset()
    base_env.base_env.unwrapped.target_jnt_value = np.array([target_angle])
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
    raw = base_env._current_raw_obs()
    obs_flat = base_env._build_obs(raw)
    if is_lstm:
        exo_env._buffer.clear()
        for _ in range(exo_env.window):
            exo_env._buffer.append(obs_flat.copy())
        no_exo_obs = exo_env._stack()
    else:
        no_exo_obs = obs_flat
    zero_action = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    no_exo_angles = []
    for _ in range(MAX_STEPS):
        no_exo_obs, _, done, truncated, _ = exo_env.step(zero_action)
        no_exo_angles.append(float(base_env.base_env.unwrapped.sim.data.qpos[0]))
        if done or truncated:
            break

    min_len_no = min(len(healthy_angles), len(no_exo_angles))
    no_exo_corr = 0.0
    if min_len_no > 1:
        no_exo_corr, _ = pearsonr(healthy_angles[:min_len_no], no_exo_angles[:min_len_no])
```

Note: `is_lstm` is already computed at the top of `run_trial` as `is_lstm = isinstance(exo_env, TemporalStackWrapper)`.

- [ ] **Step 2: Add `no_exo_correlation` to `run_trial` return dict**

In `run_trial`'s return dict (around line 202), add:

```python
        "no_exo_correlation": no_exo_corr,
```

alongside the existing `"correlation": corr`.

- [ ] **Step 3: Build no-exo matrix in `run_comparison`**

In `run_comparison`, after `trials_a` and `trials_b` are collected, add:

```python
    no_exo_trials = [{"correlation": t["no_exo_correlation"], "angle_bin": t["angle_bin"], "severity": t["severity"]} for t in trials_a]
    matrix_no_exo = build_matrix(no_exo_trials, severity_edges)
```

Note: `build_matrix` uses `t["correlation"]` to fill the matrix, so this re-uses it directly.

- [ ] **Step 4: Save no-exo confusion matrix and switch all matrices to `pct=True`**

In `run_comparison`, replace the two existing `plot_confusion_matrix` calls (around lines 412–418) with:

```python
    plot_confusion_matrix(matrix_a, ANGLE_LABELS, SEVERITY_LABELS,
                          f"Movement Accuracy (with exo) — {label_a}",
                          f"{out_dir}/confusion_matrix_{label_a}.png", pct=True)
    plot_confusion_matrix(matrix_b, ANGLE_LABELS, SEVERITY_LABELS,
                          f"Movement Accuracy (with exo) — {label_b}",
                          f"{out_dir}/confusion_matrix_{label_b}.png", pct=True)
    plot_confusion_matrix(matrix_no_exo, ANGLE_LABELS, SEVERITY_LABELS,
                          "Movement Accuracy (no exo) — impaired baseline",
                          f"{out_dir}/confusion_matrix_no_exo.png", pct=True)
```

- [ ] **Step 5: Verify `build_matrix` signature is compatible**

`build_matrix` takes `trials: list` and uses `t["correlation"]` and `t["angle_bin"]`. The `no_exo_trials` list constructed in Step 3 provides both keys. Confirm by running:

```
conda run -n exo_s python -c "
from compare import build_matrix
import numpy as np
trials = [{'correlation': 0.8, 'angle_bin': 0, 'severity': 0.3},
          {'correlation': 0.6, 'angle_bin': 1, 'severity': 0.7}]
edges = np.linspace(0, 1, 5)
m = build_matrix(trials, edges)
print('shape:', m.shape, 'ok')
"
```

Expected: `shape: (4, 4) ok`

- [ ] **Step 6: Commit**

```
git add compare.py
git commit -m "feat: add no-exo track and pct confusion matrix to compare.py"
```

---

## Task 4: Write `algo_compare.py`

**Files:**
- Create: `algo_compare.py`
- Test: `tests/test_algo_compare.py` (extend existing)

- [ ] **Step 1: Add unit tests for pure-Python scoring logic**

Add to `tests/test_algo_compare.py`:

```python
def test_compute_boost():
    """boost = how much of the gap to healthy was closed."""
    accuracy_exo   = 0.9
    accuracy_floor = 0.6
    expected = (0.9 - 0.6) / (1.0 - 0.6) * 100  # 75.0
    result = (accuracy_exo - accuracy_floor) / (1.0 - accuracy_floor) * 100
    assert abs(result - expected) < 1e-9

def test_compute_boost_floor_equals_exo():
    """If exo == floor, boost is 0%."""
    accuracy_exo   = 0.6
    accuracy_floor = 0.6
    # avoid division by zero — healthy != floor in real runs, but handle edge case
    result = (accuracy_exo - accuracy_floor) / max(1.0 - accuracy_floor, 1e-9) * 100
    assert result == 0.0

def test_pearson_perfect_correlation():
    from scipy.stats import pearsonr
    import numpy as np
    a = np.linspace(0, 2.27, 100)
    r, _ = pearsonr(a, a)
    assert r > 0.9999
```

- [ ] **Step 2: Run new tests — expect PASS (pure math, no env needed)**

```
conda run -n exo_s python -m pytest tests/test_algo_compare.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 3: Create `algo_compare.py`**

```python
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
    env.reset()
    base_env.base_env.unwrapped.target_jnt_value = np.array([target_angle])
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
    r, _ = pearsonr(a[:n], b[:n])
    return float(r)


# ---------------------------------------------------------------------------
# Track runners
# ---------------------------------------------------------------------------

def _run_healthy_track(healthy_env, healthy_policy: PPO, target_angle: float):
    """Run healthy policy to target. Returns list of qpos[0] values."""
    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = np.array([target_angle])
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
        bar = ax.bar(x[i] + w / 2, acc_exo[i], w, color=f"C{i}")
        ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                acc_exo[i] + 0.015,
                f"+{boost_pct[i]:.0f}%",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Healthy (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(policy_labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Pearson r (vs Healthy)")
    ax.set_ylim(0, 1.18)
    ax.set_title("Algorithm Accuracy Comparison — Pearson r vs Healthy Baseline")

    from matplotlib.patches import Patch
    handles, lbls = ax.get_legend_handles_labels()
    handles.append(Patch(color="steelblue", label="Impaired + exo (each policy)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

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
```

- [ ] **Step 4: Verify import structure is correct (no env instantiation)**

```
conda run -n exo_s python -c "import algo_compare; print('imports ok')"
```

Expected: `MyoSuite:> Registering Myo Envs` warnings then `imports ok`

- [ ] **Step 5: Commit**

```
git add algo_compare.py tests/test_algo_compare.py
git commit -m "feat: add algo_compare.py — cross-policy Pearson r accuracy comparison"
```

---

## Task 5: Add Stage 5 to `pipeline.py`

**Files:**
- Modify: `pipeline.py`

- [ ] **Step 1: Add `stage_algo_compare` function**

In `pipeline.py`, after the `stage_compare` function (around line 262), add:

```python
# ── Stage 5: Cross-policy accuracy comparison ─────────────────────────────

def stage_algo_compare():
    print("\n" + "=" * 60)
    print("[pipeline] STAGE 5 — Cross-policy accuracy comparison")
    print("=" * 60)

    if not os.path.exists(HEALTHY_PATH):
        print(f"  [algo_compare] Skipping — {HEALTHY_PATH} not found.")
        return

    cmd = [
        PYTHON, "algo_compare.py",
        "--healthy", HEALTHY_PATH,
    ]
    policy_flags = [
        ("--brady",         POLICY_BRADY),
        ("--deg",           POLICY_DEG),
        ("--lstm",          POLICY_LSTM),
        ("--deg-lstm",      POLICY_DEG_LSTM),
        ("--extraobs",      POLICY_EXTRAOBS),
        ("--recurrent",     POLICY_RECURRENT),
        ("--deg-recurrent", POLICY_DEG_RECURRENT),
    ]
    for flag, path in policy_flags:
        if os.path.exists(path):
            cmd += [flag, path]

    if _TEST_MODE:
        cmd += ["--episodes", "2"]

    run(cmd, "Cross-policy accuracy comparison")
```

- [ ] **Step 2: Call `stage_algo_compare()` in `main()`**

In `pipeline.py`'s `main()`, after the `stage_compare()` call, add:

```python
    stage_algo_compare()
```

- [ ] **Step 3: Add output paths to the final summary print block**

In `main()`, after the existing `print("  results/recurrentppo_vs_lstm_ablation/")` line, add:

```python
    print("  results/algo_compare/accuracy_bar.png")
    print("  results/algo_compare/accuracy_per_episode.png")
```

- [ ] **Step 4: Verify `pipeline.py` parses without error**

```
conda run -n exo_s python -c "import pipeline; print('ok')"
```

Expected: `ok` (no import errors)

- [ ] **Step 5: Commit**

```
git add pipeline.py
git commit -m "feat: add Stage 5 (algo_compare) to pipeline.py"
```

---

## Self-Review

### Spec Coverage

| Spec section | Task |
|---|---|
| 100 shared seeds, same patient per episode | Task 4 — `_sample_patient_state` with `np.random.RandomState(seed)` |
| Healthy = baseline (Pearson r ≈ 1.0) | Task 4 — `_run_healthy_track` |
| No-exo track (zero action, impaired env) | Task 4 — `_run_no_exo_track` |
| Pearson r accuracy metric | Task 4 — `_pearsonr_safe` |
| Boost % formula | Task 4 — `gap = max(1.0 - no_exo_mean, 1e-9)` |
| Grouped bar chart with annotations | Task 4 — `_plot_bar` |
| Line plot one-line-per-policy | Task 4 — `_plot_line` |
| `pct` mode in `plot_confusion_matrix` | Task 1 |
| `evaluation.py` no-exo matrix | Task 2 |
| `compare.py` no-exo track + matrix | Task 3 |
| Stage 5 in `pipeline.py` | Task 5 |
| MyoSuite API: `np.array([target_angle])`, `target_type="fixed"`, `update_target(restore_sim=True)` | Tasks 3, 4 |
| `max_episode_steps=100` respected via `if done or truncated: break` | All tracks |

### No Placeholders

All code blocks are complete. No "TBD" or "similar to Task N".

### Type Consistency

- `_configure_exo_env` returns obs (array) — used as initial obs in `_run_no_exo_track` and `_run_exo_track`. ✓
- `_sample_patient_state` returns 6 values in a fixed order — called identically in Phase 1 and Phase 2 with the same `np.random.RandomState(seed)`. ✓
- `build_matrix` in `compare.py` expects `t["correlation"]` and `t["angle_bin"]` — `no_exo_trials` provides both. ✓
- `plot_confusion_matrix(pct=True)` — called with keyword arg everywhere. ✓
