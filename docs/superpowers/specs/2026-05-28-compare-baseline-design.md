# Design: Baseline vs RecPPO Policy Comparison (`compare_baseline.py`)

**Date:** 2026-05-28
**Status:** Approved
**Scope:** New `compare_baseline.py` → outputs to `results/baseline_comparison/`

---

## 1. Goal

Produce a fair head-to-head comparison between:
- **MLP deg-only baseline** (`policy_deg.zip`) — trained WITHOUT bradykinesia, evaluated on the brady+deg environment (out-of-distribution for it)
- **RecPPO brady+deg** (`policy_brady_deg_recurrent.zip`) — trained WITH bradykinesia and degradation, evaluated on the same environment

The key question: how much does the RecPPO policy improve over a baseline that was never trained for bradykinesia?

---

## 2. Policy Paths (Hardcoded)

```python
HEALTHY_PATH   = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\healthy_policy.zip"
BASELINE_PATH  = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\policy_deg.zip"
RECURRENT_PATH = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\policy_brady_deg_recurrent.zip"
OUT_DIR        = "results/baseline_comparison"
N_EPISODES     = 80
```

---

## 3. Environment Setup

Both exo policies are evaluated in the **brady+deg** environment:

```python
CombinedExoOnlyWrapper(
    base_env,
    frozen_policy_path=HEALTHY_PATH,
    bradykinesia=True,
    smart_reset=True,
    hide_pose_err=True,
    extra_obs=False,
)
```

- `policy_deg` is plain PPO (MLP), not recurrent, no LSTM wrapper needed
- `policy_brady_deg_recurrent` is RecurrentPPO — use `is_recurrent=True` flag in predict loop

---

## 4. Episode Plan (80 Episodes, 5 Per Cell)

Uses `plan_trials(80)` from `evaluation.py` (already imported via module reuse):

```
4 angle bins × 4 severity quartiles = 16 cells
80 / 16 = 5 episodes per cell (remainder 0 — perfect fit)
```

Episode plan is generated once. Both policies run on the exact same 80 episodes in the same order. Shared patient state is achieved by fixing the same `target_angle`, `force_scale`, `activation_slowdown`, and `MF/MA/MR` values for every episode.

**Per-episode sampling:**
- Target angle: sampled uniformly within the angle bin range
- Patient state: sampled within the severity quartile ranges using `severity_quartile_to_range()`
- Same RNG seed per episode cell assignment (deterministic episode plan)

---

## 5. Three Tracks Per Episode

1. **Healthy track** — `healthy_policy` on `myoElbowPose1D6MRandom-v0`, same `target_angle`, qpos=0 start
2. **Baseline track** — `policy_deg` on brady+deg env, same patient state as track 3
3. **RecPPO track** — `policy_brady_deg_recurrent` on brady+deg env, same patient state

Tracks 2 and 3 share the same `target_angle`, `force_scale`, `activation_slowdown`, `MF`, `MA`, `MR` values (configured explicitly after env reset, mirroring the pattern in `evaluation.py:run_eval_trial`).

A fourth **no-exo track** (zero action, same patient state) is also run for the confusion matrix floor.

---

## 6. Metrics Per Episode

```python
baseline_corr  = pearsonr(baseline_angles[:min_len],  healthy_angles[:min_len])
recurrent_corr = pearsonr(recurrent_angles[:min_len], healthy_angles[:min_len])
no_exo_corr    = pearsonr(no_exo_angles[:min_len],    healthy_angles[:min_len])
baseline_reward  = sum of step rewards
recurrent_reward = sum of step rewards
```

Angle bin and severity quartile are recorded per episode for confusion matrix placement.

---

## 7. Confusion Matrices

Three 4×4 matrices (angle bins × severity quartiles), Pearson r averaged per cell:

| File | Content |
|---|---|
| `confusion_matrix_baseline.png` | MLP deg-only on brady+deg env |
| `confusion_matrix_recppo.png` | RecPPO on brady+deg env |
| `confusion_matrix_no_exo.png` | No-exo floor (zero action) |

Style matches existing results: Blues colormap, `pct=True`, cells labeled as `%`, using `plot_confusion_matrix` from `utils.py`.

Labels:
- Angle: `["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5"]`
- Severity: `["Q1 mild", "Q2", "Q3", "Q4 severe"]`

---

## 8. Outputs

```
results/baseline_comparison/
  confusion_matrix_baseline.png    # 4×4 Pearson r — MLP deg-only
  confusion_matrix_recppo.png      # 4×4 Pearson r — RecPPO
  confusion_matrix_no_exo.png      # 4×4 Pearson r — no-exo floor
  comparison_metrics.png           # 4-panel combined figure
  pearsonr_by_severity.png         # Pearson r sorted by severity (separate PNG)
```

---

## 9. 4-Panel Combined Figure (`comparison_metrics.png`)

Single figure, 2×2 grid of subplots, ~14×10 inches, 150 dpi.

**Panel 1 (top-left): Reward per Episode**
- X: Episode 1–80 (sequential)
- Y: Cumulative episode reward
- Two lines: baseline (steelblue) and RecPPO (coral)
- Horizontal dashed lines at mean reward for each
- Legend

**Panel 2 (top-right): Mean Pearson r Bar Chart**
- Three bars: Baseline, RecPPO, No-Exo Floor
- Colors: steelblue, coral, gray
- Y: Pearson r (0–1)
- Dashed line at y=1.0 ("Healthy")
- Boost % annotation on each bar relative to no-exo floor
- Y-axis label: "Mean Pearson r (vs Healthy)"

**Panel 3 (bottom-left): Pearson r Per Episode — Sequential**
- X: Episode 1–80
- Y: Pearson r
- Two lines (baseline + RecPPO) + shaded area between them showing the gap
- Dashed line at y=1.0 (Healthy), dotted line at mean no-exo floor
- Legend

**Panel 4 (bottom-right): Pearson r Sorted by Severity**
- X: Episodes re-sorted from mild (Q1) to severe (Q4) severity
- Y: Pearson r
- Same two lines + shading
- X-axis tick labels mark severity quartile boundaries
- Shows how each policy degrades as patient condition worsens

---

## 10. Separate Severity-Sorted Plot (`pearsonr_by_severity.png`)

Same as Panel 4 but as a standalone larger figure (~12×5 inches) with more detail — useful for individual inclusion in a paper/report.

---

## 11. Console Summary

```
============================================================
Baseline Comparison Summary
  Episodes        : 80
  Baseline policy : policy_deg (MLP, deg-only trained)
  RecPPO policy   : policy_brady_deg_recurrent

  Mean Pearson r  (Baseline) : 0.xxx
  Mean Pearson r  (RecPPO)   : 0.xxx
  Mean Pearson r  (No-exo)   : 0.xxx
  Boost Baseline             : +xx.x% toward healthy
  Boost RecPPO               : +xx.x% toward healthy
  RecPPO advantage           : +xx.x% over baseline
============================================================
```

---

## 12. What Is NOT Changed

- `evaluation.py` — untouched
- `compare.py` — untouched
- `algo_compare.py` — untouched
- `utils.py` — untouched (only imported)
- `envs/elbow_env.py` — untouched (only imported)
- No new training; policies used as-is

---

## 13. Reused Imports

From `evaluation.py` (imported as module functions, not copy-pasted):
- `plan_trials(n)`
- `severity_quartile_to_range(quartile)`
- `angle_bin_to_target(bin, edges)`

From `utils.py`:
- `plot_confusion_matrix(..., pct=True)`
- `compute_severity(...)`
- `get_angle_bin(...)`
- `get_severity_quartile(...)`
