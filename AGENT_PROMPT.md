# ReGainX Agent Task Brief

You are working inside the ReGainX codebase (https://github.com/TerminalRocketship45/ReGainX).
This is a Parkinson's disease elbow exoskeleton RL project built on MyoSuite + Stable-Baselines3.
All scripts are already written. Your job is to run them in order and return structured results
so the paper author can paste numbers directly into LaTeX tables.

---

## Environment Setup

```bash
conda activate exo
# Verify scipy is available (needed for significance tests):
python -c "from scipy.stats import ttest_rel, wilcoxon; print('scipy OK')"
```

All policies live in `policies/`. The evaluation results go to `results/shared_eval/`.

---

## Step 1 — Shared Evaluation (if not already done)

Check whether `results/shared_eval/raw/policy_brady_deg_recurrent_trials.csv` exists.

**If it already exists:** skip to Step 2.

**If it does not exist**, run the full shared evaluation first (~30–60 min, 1 GPU):
```bash
python run_shared_evaluation.py --episodes 4000 --seed 42
```

This evaluates all 6 RL policies on 4,000 identical episodes stratified across
severity quartiles (Q1–Q4) and radian bins. Outputs go to `results/shared_eval/`.

---

## Step 2 — Statistical Significance Tests

```bash
python run_stat_significance.py --latex
```

This reads the existing per-episode CSVs and runs:
- Paired two-sided t-test
- Wilcoxon signed-rank test

…for every policy pair (BD RecPPO vs each ablation) on Pearson r and goal rate,
including a per-severity-quartile breakdown.

---

## Step 3 — Impedance Baseline (~30–45 min, 1 GPU)

```bash
python run_impedance_baseline.py
```

This runs a velocity-deficit impedance controller (Gasperina et al., 2021)
on the same 4,000 episodes. It:
- Runs the healthy reference policy to collect target trajectories
- Runs the impedance controller on all episodes
- Computes Pearson r, goal rate, and reward per episode and per quartile
- Runs significance tests vs BD RecPPO
- Generates two comparison figures
- Prints LaTeX table rows

---

## What to Return

Return ALL of the following in a single structured response. Use the exact
section labels below so the author can find them easily.

---

### SECTION 1 — Per-Policy Per-Severity Table (fills Table A1 in paper appendix)

Paste the printed summary from `run_shared_evaluation.py` OR read from:
`results/shared_eval/summaries/*_per_quartile.csv`

Format (fill every cell — no dashes):
```
Policy               | Q1 r  | Q2 r  | Q3 r  | Q4 r  | Q1 goal | Q2 goal | Q3 goal | Q4 goal
No exoskeleton       |       |       |       |       |         |         |         |
Impedance Baseline   |       |       |       |       |         |         |         |
Deg MLP              |       |       |       |       |         |         |         |
BD MLP               |       |       |       |       |         |         |         |
Deg RecPPO           |       |       |       |       |         |         |         |
BD RecPPO            |       |       |       |       |         |         |         |
Noisy RecPPO         |       |       |       |       |         |         |         |
```

Also include mean reward per quartile for Deg RecPPO and BD RecPPO (already in paper
Table 2, but confirm numbers match).

---

### SECTION 2 — Statistical Significance Table (fills Table A2 in paper appendix)

Copy the exact output from `python run_stat_significance.py --latex`, which prints
ready-to-paste LaTeX rows. Also paste the human-readable table.

Required rows:
- BD RecPPO vs No-exo         (Pearson r)
- BD RecPPO vs Deg RecPPO     (Pearson r)
- BD RecPPO vs BD MLP         (Pearson r)
- BD RecPPO vs Deg MLP        (Pearson r)
- BD RecPPO vs Noisy RecPPO   (Pearson r)
- BD RecPPO vs Impedance      (Pearson r)
- BD RecPPO vs No-exo         (Goal rate)
- BD RecPPO vs Deg RecPPO     (Goal rate)
- BD RecPPO vs Impedance      (Goal rate)

For each: n, mean_a, mean_b, Δ, t-test p, Wilcoxon p, significance stars.

---

### SECTION 3 — Per-Radian Table (fills Table A3 in paper appendix)

Read from:
- `results/shared_eval/summaries/policy_brady_deg_recurrent_per_radian.csv`
- `results/shared_eval/summaries/policy_deg_recurrent_per_radian.csv`
- `results/shared_eval/summaries/impedance_baseline_per_radian.csv`

Format:
```
Radian bin   | BD r  | BD goal | Deg r | Deg goal | Imp r | Imp goal
0.0–0.5 rad  |       |         |       |          |       |
0.5–1.0 rad  |       |         |       |          |       |
1.0–1.5 rad  |       |         |       |          |       |
1.5+ rad     |       |         |       |          |       |
```

---

### SECTION 4 — Impedance Baseline Summary (narrative + per-quartile)

Copy the printed console output from `run_impedance_baseline.py`:
- Overall Pearson r ± std
- Goal achievement rate
- Mean episode reward
- Per-quartile breakdown (Q1–Q4)
- Significance vs BD RecPPO (t-test p, Wilcoxon p)

---

### SECTION 5 — LaTeX Rows (direct paste)

Paste the `--latex` output from both scripts:
1. From `run_stat_significance.py --latex`
2. From `run_impedance_baseline.py` (the "LaTeX rows for Table A1/A2" section)

---

### SECTION 6 — Figure Paths

Confirm these files exist and report their sizes:
- `results/shared_eval/plots/comparison_pearsonr_with_impedance.png`
- `results/shared_eval/plots/pearsonr_by_severity_with_impedance.png`

---

### SECTION 7 — Issues / Anomalies

Report any:
- Policies not found (skipped)
- Episode counts that differ from expected 4,000
- NaN values in outputs
- Any script errors and how they were resolved

---

## Notes

- The impedance controller is in `envs/impedance_baseline.py`. Do NOT modify it.
- The significance script is in `run_stat_significance.py`. Do NOT modify it.
- All CSVs use `trial_idx` as the shared key for episode matching across policies.
- The healthy reference is re-run by `run_impedance_baseline.py` — this is intentional
  (it ensures matched starting conditions for the Pearson r calculation).
- If `run_shared_evaluation.py` was previously run with `--skip-existing`, the raw CSVs
  should already be present and Step 1 can be skipped entirely.
