# Design: Algorithm Accuracy Comparison (`algo_compare`)

**Date:** 2026-05-24  
**Status:** Approved  
**Scope:** New `algo_compare.py` + confusion matrix fix in `evaluation.py` / `compare.py` / `utils.py` + Stage 5 in `pipeline.py`

---

## 1. Goal

Produce a single, fair cross-policy accuracy comparison that answers: "How close does each exoskeleton policy bring the impaired patient to healthy movement?"

The output is two charts in `results/algo_compare/`, generated automatically by `python pipeline.py`.

---

## 2. Accuracy Metric

**Primary metric: Pearson correlation** between the exo-assisted trajectory and the healthy reference trajectory.

- **Healthy = 1.0 (100%)** — healthy policy on the clean `myoElbowPose1D6MRandom-v0` env, no fatigue, no bradykinesia, no exoskeleton.
- **Impaired no-exo** — same patient state, zero exo torque (zero action). This is the floor.
- **Impaired + exo** — impaired env with the exo policy active. This is the metric for each model.

Per episode:
```
r_healthy  ≈ 1.0          (healthy vs itself)
r_no_exo   = pearsonr(no_exo_angles, healthy_angles)
r_exo[p]   = pearsonr(exo_angles[p], healthy_angles)
```

Summary:
```
accuracy[p]    = mean(r_exo[p])      over 100 episodes
accuracy_floor = mean(r_no_exo)      over 100 episodes  (same for all models)
boost[p]       = (accuracy[p] - accuracy_floor) / (1.0 - accuracy_floor) * 100%
```

---

## 3. Episode Setup (100 Shared Seeds)

Seeds generated once per run:
```python
seeds = np.random.default_rng(42).integers(0, 2**31, size=100)
```

Per episode (seed `s`):
```python
rng = np.random.RandomState(s)
target_angle      = rng.uniform(0.0, 2.27)          # r_elbow_flex range from MyoSuite source
force_scale       = rng.uniform(0.6, 0.9)            # bradykinesia force scaling
activation_slow   = rng.uniform(1.1, 1.4)            # bradykinesia slowdown
MF                = rng.uniform(0.7, 1.0, n_muscles) # muscle fatigue
```

**MyoSuite API notes (confirmed from installed source):**
- `target_jnt_value` is `np.array([angle])` (1-element array)
- Target-locking sequence: set `target_jnt_value`, set `target_type = "fixed"`, call `update_target(restore_sim=True)`
- `rwd_dict["solved"]` = `|target_jnt_value - qpos[0]| < 0.175` rad
- `max_episode_steps = 100` (gym truncates automatically); `MAX_STEPS = 200` used as loop guard
- `qpos[0]` = `r_elbow_flex` joint angle

**Three tracks per episode:**

1. **Healthy track** — `myoElbowPose1D6MRandom-v0`, healthy policy, same `target_angle`
2. **No-exo track** — `CombinedExoOnlyWrapper` with same patient state, `zero_action = np.zeros(exo_env.action_space.shape)`
3. **Exo track (per policy)** — same env + patient state, exo policy active

All three tracks start from `qpos = 0` (arm fully extended) after target is set.

---

## 4. New File: `algo_compare.py`

**CLI:**
```
python algo_compare.py \
  --healthy    policies/healthy_policy.zip \
  --brady      policies/policy_brady_deg.zip \
  --deg        policies/policy_deg.zip \
  --lstm       policies/policy_brady_deg_lstm.zip \
  --deg-lstm   policies/policy_deg_lstm.zip \
  --extraobs   policies/policy_brady_deg_lstm_extraobs.zip \
  --recurrent  policies/policy_brady_deg_recurrent.zip \
  --deg-recurrent policies/policy_deg_recurrent.zip \
  [--episodes 100] [--seed 42]
```

Missing policy paths are silently skipped (same pattern as existing pipeline stages).

**Internal flow:**
1. Load all provided policies
2. Build one `healthy_env`, one `exo_env` (reused across all episodes)
3. Generate 100 seeds
4. Run the 3-track loop: healthy → no-exo → each exo policy
5. Collect per-episode `r_no_exo` and `r_exo[policy]`
6. Compute summary scores and boost %
7. Save two charts

**Outputs:**
```
results/algo_compare/accuracy_bar.png
results/algo_compare/accuracy_per_episode.png
```

---

## 5. Bar Chart (`accuracy_bar.png`)

- One group per exo model (up to 7 models)
- Each group: two bars
  - **Gray** — no-exo Pearson r (same height for all groups — shared patient conditions)
  - **Colored** — with-exo Pearson r
- Y-axis: Pearson r (0–1.0), labeled "Pearson r (vs Healthy)"
- Horizontal dashed line at `y = 1.0` labeled "Healthy (100%)"
- Boost annotation on each with-exo bar: `+{boost:.0f}% toward healthy`
- Legend, tight layout, 150 dpi, saved to `results/algo_compare/`

---

## 6. Line Plot (`accuracy_per_episode.png`)

- X-axis: Episode 1–100
- Y-axis: Pearson r per episode
- One line per exo policy (distinct colors from matplotlib `C0`–`C6`)
- Horizontal dashed line at `y ≈ 1.0` labeled "Healthy"
- Horizontal dotted gray line at `y = mean(r_no_exo)` labeled "No-exo floor"
- Legend, tight layout, 150 dpi

---

## 7. Confusion Matrix Fix

**What changes:** The existing confusion matrices in `evaluation.py` and `compare.py` only show the with-exo Pearson r. They are updated to:
1. Also compute and save a **no-exo confusion matrix** using `no_exo_angles` (already collected by `evaluation.py`)
2. Express values as **percentage** (×100, cells labeled `%`) to match the accuracy framing

**`utils.py` — `plot_confusion_matrix` signature change:**
```python
def plot_confusion_matrix(matrix, angle_labels, severity_labels, title, save_path, pct=False):
```
When `pct=True`: multiply display values by 100, label cells as `{val:.0f}%`, retitle colorbar as `"% trajectory match vs healthy"`, set `vmax=100`.

**`evaluation.py` changes:**
- Track `no_exo_angles` per trial (already done — `no_exo_angles` is collected)
- Build a `matrix_no_exo` from `no_exo_angles` vs `healthy_angles` correlation
- Save `confusion_matrix_no_exo.png` and `confusion_matrix_with_exo.png` (rename existing `confusion_matrix.png` → `confusion_matrix_with_exo.png`)
- Pass `pct=True` to both matrix plots

**`compare.py` changes:**
- Same: save `confusion_matrix_{label_a}_no_exo.png` and existing `confusion_matrix_{label_a}.png` with `pct=True`
- The no-exo matrix uses trials from the no-exo track (needs a `run_trial_no_exo` variant or inline zero-action roll-out per ablation comparison)

---

## 8. Pipeline Integration

New Stage 5 added to `pipeline.py` after Stage 4:

```python
def stage_algo_compare():
    cmd = [PYTHON, "algo_compare.py",
           "--healthy", HEALTHY_PATH,
           "--brady",   POLICY_BRADY, ...]
    # skip missing policies
    run(cmd, "Algorithm accuracy comparison")
```

Prints added to the final summary block:
```
results/algo_compare/accuracy_bar.png
results/algo_compare/accuracy_per_episode.png
```

---

## 9. What Is NOT Changed

- `train_*.py` scripts — untouched
- `envs/elbow_env.py` — untouched
- `envs/temporal_buffer.py` — untouched
- Existing ablation comparison logic in `compare.py` — only the confusion matrix plotting call and no-exo matrix addition

---

## 10. MyoSuite Source References

Confirmed from `site-packages/myosuite/envs/myo/myobase/__init__.py` and `pose_v0.py`:

| Fact | Value |
|---|---|
| `r_elbow_flex` range | [0.0, 2.27] rad |
| `pose_thd` (solved threshold) | 0.175 rad |
| `max_episode_steps` | 100 |
| `target_jnt_value` type | `np.array([float])` |
| Target-lock API | set `target_jnt_value`, set `target_type="fixed"`, call `update_target(restore_sim=True)` |
| `solved` in `rwd_dict` | `pose_dist < pose_thd` where `pose_dist = norm(target_jnt_value - qpos)` |
| Fatigue env | Auto-registered from exo env with `muscle_condition: "fatigue"` |
