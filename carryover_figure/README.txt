ReGainX fatigue-carryover experiment — results
==============================================

Contents
--------
  fig_carryover.pdf / .png   Two-panel task-200 unassisted-reach figure.
  carryover_log.csv          Per-(seed,condition,task) metrics; 30*2*200 rows.
  carryover_task200.npz       Per-seed task-200 joint-angle trajectories (A,B),
                              healthy reference, target — used to build the fig.
  carryover_summary.txt       The Step-3 report (numbers below).
  (source scripts at repo root: run_carryover_experiment.py,
   make_carryover_figure.py)

How it was produced
-------------------
  D:\exo_s\python.exe run_carryover_experiment.py --seeds 30
  D:\exo_s\python.exe make_carryover_figure.py
Interpreter: Python 3.12.7 at D:\exo_s (myosuite + sb3_contrib + mujoco).
Every number is read back from actually stepping the MyoSuite
`myoFatiElbowPose1D6MExoRandom-v0` simulation. Nothing is hand-set to create
the effect.

Design
------
PD patient, moderate-severe: force_scale=0.65, activation_slowdown=1.3,
HIGH initial muscle-fatigue fraction MF=0.70. 200 consecutive reaching tasks
(fixed start=0, target=2.0 rad), 100 steps each, 30 seeds.
  A (no assistance):    exo torque = 0 for all 200 tasks.
  B (assisted history): RecurrentPPO brady+deg exo assists tasks 1-199; the exo
                        is REMOVED for task 200. BOTH patients do task 200
                        unassisted — the only difference is the fatigue each
                        accumulated over tasks 1-199.

STEP 1 — fatigue accumulation (verified in code AND by probe)
-------------------------------------------------------------
The 3CC-r model (CumulativeFatigue, 6 muscles) accumulates fatigue within a
task as muscles activate, but CombinedExoOnlyWrapper.reset()
(envs/elbow_env.py:214-248) wipes it every episode: base_env.reset(
fatigue_reset=True) plus smart_reset overwriting MF ~ U[0.7,1.0]. So by default
fatigue does NOT carry across tasks. Probe confirmed: MF 0.05->0.055 over 100
steps (accumulates), then 0.815 after reset (wiped).

MODIFICATION (what we changed to enable carryover)
--------------------------------------------------
The wrapper is not edited. run_carryover_experiment.py persists fatigue in the
run loop: seed MF=0.70 at task 0; after each env.reset() RESTORE the
muscle_fatigue.{MA,MR,MF} arrays saved at the previous task's end (overriding
smart_reset); step the task so the sim accumulates fatigue physically; SAVE
{MA,MR,MF} for the next task. Fatigue is thus continuous across all 200 tasks.
The exo's only effect on fatigue is indirect: supplying torque changes how hard
the patient's own muscles must work, changing accumulation. Probe Test C
confirmed save/restore reproduces state exactly and it keeps accumulating.

STEP 3 — RESULTS (mean +/- std over 30 seeds)
---------------------------------------------
Cumulative fatigue (avg MF) entering task 200:
  A (no assist)      : 0.8254 +/- 0.0002
  B (assisted 1-199) : 0.4820 +/- 0.0193
  reduction B vs A   : 41.6%
Task-200 unassisted performance:
  final angle (rad): A = -0.000 +/- 0.000   B = 0.018 +/- 0.025
  Pearson r        : A = -0.782 +/- 0.033   B = 0.310 +/- 0.627
  goal success     : A = 0.000              B = 0.000
Goal rate over last 50 tasks: A = 0.000     B = 0.451

HONEST INTERPRETATION (important)
---------------------------------
The carryover effect on FATIGUE is large and real: 199 tasks of prior exo
assistance left the muscles 41.6% less fatigued (MF 0.48 vs 0.83) entering the
unassisted task 200, because the exo did the work and the muscles recovered
below the 0.70 start, while the no-assistance patient strained to near-
saturation.

However, at this severity the UNASSISTED impaired arm barely moves in EITHER
condition: task-200 mean reach is -0.00 rad (A) vs 0.018 rad (B), and NEITHER
reaches the 2.0 rad target (goal success 0 for both). The exoskeleton, not the
fatigue state, is what enables reaching at force_scale=0.65; remove it and the
arm cannot complete the reach regardless of fatigue history. The prior-exo
patient does move slightly more and tracks the healthy trajectory shape better
(Pearson r 0.31 vs -0.78, i.e. the no-history arm drifts backward), but the
difference in absolute reach is small.

The figure reflects exactly this: the main axes show both conditions falling
far short of the 2.0 rad target line; the zoomed insets show the small but real
reach difference; the annotations report the accumulated-fatigue values. No
label claims the target was reached, and the words months/recovery/cured/
therapy/retention appear nowhere.
