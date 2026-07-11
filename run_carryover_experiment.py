"""
ReGainX fatigue-carryover experiment (Steps 1-6).

HONESTY / PROVENANCE
--------------------
Every number this produces comes from actually stepping the MyoSuite
`myoFatiElbowPose1D6MExoRandom-v0` simulation. Nothing is hand-set to create
the effect. The ONLY intervention is making muscle fatigue PERSIST across
consecutive tasks (documented below) — the fatigue difference between the two
conditions must then emerge from the simulation itself.

  >>> This script had NOT been executed at the time it was written, because the
  >>> host environment was blocking launch of the Python interpreter
  >>> (STATUS_ACCESS_DENIED on python.exe). Run it yourself once Python is
  >>> runnable; do not trust any numbers until this has actually run. <<<

STEP 1 FINDING (from envs/elbow_env.py, verified by reading source)
-------------------------------------------------------------------
`CombinedExoOnlyWrapper.reset()` (lines 214-248) does two things that DESTROY
carryover every episode:
  1. calls `base_env.reset(fatigue_reset=True)`  -> myosuite zeroes fatigue
  2. `smart_reset` overwrites MF with np.random.uniform(0.7, 1.0) each reset
Within a single task the 3CC-r model DOES accumulate fatigue as muscles
activate (base_env.step integrates MA/MR/MF), but that accumulation is thrown
away at the next reset. So fatigue RESETS per episode.

THE MODIFICATION (what we changed, and it is the whole point)
-------------------------------------------------------------
We do not edit the wrapper. Instead the run loop persists fatigue itself:
  * At task 0 we seed a HIGH initial fatigue (MF = INIT_MF ~ 0.7).
  * We call env.reset() to start each task (new pose/target), then IMMEDIATELY
    RESTORE the muscle_fatigue.{MA,MR,MF} arrays saved at the end of the
    previous task, overriding the wrapper's smart_reset randomisation.
  * We run the task; the sim accumulates fatigue physically during stepping.
  * At task end we SAVE muscle_fatigue.{MA,MR,MF} to carry into the next task.
This makes fatigue a continuous state across all 200 tasks. The exoskeleton's
only effect on fatigue is indirect: by supplying torque it changes how hard the
patient's own muscles must work, which changes how fatigue accumulates.

CONDITIONS
----------
A (no assistance):     exo torque = 0 for all 200 tasks.
B (assisted history):  exo policy drives torque for tasks 1..199, then the exo
                       is REMOVED for task 200 (torque = 0). BOTH patients do
                       task 200 unassisted; the only difference is the fatigue
                       accumulated over tasks 1..199.

Outputs (into carryover_figure/):
  carryover_log.csv            per (seed, condition, task) metrics
  carryover_task200.npz        per-seed task-200 joint-angle trajectories (A,B)
  carryover_summary.txt        the Step-3 report
"""

import os
import argparse
import numpy as np

import myosuite  # noqa: F401  (registers envs)
from myosuite.utils import gym
from scipy.stats import pearsonr
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from envs.elbow_env import CombinedExoOnlyWrapper

# ----------------------------------------------------------------------
# Fixed experiment configuration
# ----------------------------------------------------------------------
HEALTHY_PATH = "policies/healthy_policy.zip"
EXO_PATH = "policies/policy_brady_deg_recurrent.zip"   # RecurrentPPO brady+deg

N_TASKS = 200
STEPS_PER_TASK = 100
TARGET_ANGLE = 2.0            # rad; identical start(=0)/target for every task
INIT_MF = 0.70               # HIGH initial muscle-fatigue fraction (Step 2)
FORCE_SCALE = 0.65           # moderate-severe PD
ACT_SLOWDOWN = 1.3
GOAL_TOL = 0.10              # |angle - target| considered "reached" (fallback)
OUT_DIR = "carryover_figure"


def build_envs():
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    exo_env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_PATH,
        bradykinesia=True,
        smart_reset=True,                       # we override its randomisation
        hide_pose_err=True,
        extra_obs=False,
        force_scale_range=(FORCE_SCALE, FORCE_SCALE),
        activation_slowdown_range=(ACT_SLOWDOWN, ACT_SLOWDOWN),
    )
    healthy_env = gym.make("myoElbowPose1D6MRandom-v0")
    healthy_policy = PPO.load(HEALTHY_PATH)
    exo_policy = RecurrentPPO.load(EXO_PATH)
    return exo_env, healthy_env, healthy_policy, exo_policy


def start_task(exo_env, restore_state):
    """Begin a task: reset, set fixed target + zero pose, restore fatigue.

    `exo_env` is the CombinedExoOnlyWrapper; the raw myosuite env is
    `exo_env.base_env.unwrapped` (fatigue/sim live there).
    """
    u = exo_env.base_env.unwrapped
    exo_env.reset()
    u.target_jnt_value = [TARGET_ANGLE]
    u.target_type = "fixed"
    u.update_target(restore_sim=True)
    # persist fatigue: override smart_reset randomisation with carried state
    u.muscle_fatigue.MA[:] = restore_state["MA"]
    u.muscle_fatigue.MR[:] = restore_state["MR"]
    u.muscle_fatigue.MF[:] = restore_state["MF"]
    # bradykinesia (force/activation) already applied by reset(); ensure fixed
    exo_env.force_scale = FORCE_SCALE
    exo_env.activation_slowdown = ACT_SLOWDOWN
    exo_env._apply_brady()
    u.sim.data.qpos[:] = 0.0
    u.sim.data.qvel[:] = 0.0
    u.sim.forward()


def save_state(exo_env):
    mf = exo_env.base_env.unwrapped.muscle_fatigue
    return {"MA": mf.MA.copy(), "MR": mf.MR.copy(), "MF": mf.MF.copy()}


def healthy_reference(healthy_env, healthy_policy):
    """Deterministic healthy trajectory to the fixed target (reused for r)."""
    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = [TARGET_ANGLE]
    healthy_env.unwrapped.target_type = "fixed"
    healthy_env.unwrapped.update_target(restore_sim=True)
    healthy_env.unwrapped.sim.data.qpos[:] = 0.0
    healthy_env.unwrapped.sim.data.qvel[:] = 0.0
    healthy_env.unwrapped.sim.forward()
    obs_dim = healthy_policy.observation_space.shape[0]
    obs = healthy_env.unwrapped.get_obs()[:obs_dim]
    angles = []
    for _ in range(STEPS_PER_TASK):
        a, _ = healthy_policy.predict(obs, deterministic=True)
        nobs, _, done, trunc, _ = healthy_env.step(a)
        obs = nobs[:obs_dim]
        angles.append(float(healthy_env.unwrapped.sim.data.qpos[0]))
        if done or trunc:
            break
    return np.array(angles)


def run_task(exo_env, exo_policy, assisted, healthy_angles):
    """Step one task. Returns per-task metrics + full angle trajectory."""
    u = exo_env.base_env.unwrapped
    obs = exo_env._build_obs(exo_env._current_raw_obs())
    lstm_states, ep_start = None, np.ones((1,), dtype=bool)
    angles, summed_act = [], 0.0
    goal = False
    for _ in range(STEPS_PER_TASK):
        if assisted:
            action, lstm_states = exo_policy.predict(
                obs, state=lstm_states, episode_start=ep_start, deterministic=True)
            ep_start = np.zeros((1,), dtype=bool)
        else:
            action = np.zeros(exo_env.action_space.shape, dtype=np.float32)
        obs, _, done, trunc, _ = exo_env.step(action)
        angles.append(float(u.sim.data.qpos[0]))
        summed_act += float(np.sum(u.sim.data.act))  # summed muscle activation
        solved = u.rwd_dict.get("solved", False)
        if bool(np.asarray(solved).flat[0]):
            goal = True
        if done or trunc:
            break
    angles = np.array(angles)
    m = min(len(healthy_angles), len(angles))
    r = pearsonr(healthy_angles[:m], angles[:m])[0] if m > 1 else np.nan
    if not goal:  # fallback goal definition if rwd_dict lacks 'solved'
        goal = bool(len(angles) and abs(angles[-1] - TARGET_ANGLE) <= GOAL_TOL)
    return {
        "final_angle": float(angles[-1]) if len(angles) else np.nan,
        "pearson_r": float(r),
        "summed_activation": summed_act,
        "goal": bool(goal),
        "trajectory": angles,
    }


def run_condition(seed, condition, exo_env, exo_policy, healthy_angles):
    """Run all 200 tasks for one condition under one seed. Persistent fatigue."""
    u = exo_env.base_env.unwrapped
    np.random.seed(seed)
    n = exo_env.n_muscles
    # HIGH initial fatigue, physiologically valid split (MA+MR+MF = 1)
    MF = np.full(n, INIT_MF)
    split = np.random.uniform(0.0, 1.0, size=n)
    state = {"MF": MF, "MA": (1 - MF) * split, "MR": (1 - MF) * (1 - split)}

    rows, task200_traj = [], None
    for t in range(1, N_TASKS + 1):
        assisted = (condition == "B") and (t <= N_TASKS - 1)  # exo off for task 200
        start_task(exo_env, state)
        mf_start = float(np.mean(u.muscle_fatigue.MF))
        res = run_task(exo_env, exo_policy, assisted, healthy_angles)
        state = save_state(exo_env)  # carry fatigue forward
        rows.append({
            "seed": seed, "condition": condition, "task": t,
            "assisted": int(assisted), "mf_start": mf_start,
            "summed_activation": res["summed_activation"],
            "pearson_r": res["pearson_r"], "goal": int(res["goal"]),
            "final_angle": res["final_angle"],
        })
        if t == N_TASKS:
            task200_traj = res["trajectory"]
    return rows, task200_traj


def main():
    global N_TASKS, STEPS_PER_TASK
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=N_TASKS,
                    help="override number of tasks (smoke testing)")
    ap.add_argument("--steps", type=int, default=STEPS_PER_TASK,
                    help="override steps per task (smoke testing)")
    args = ap.parse_args()
    N_TASKS, STEPS_PER_TASK = args.tasks, args.steps
    os.makedirs(OUT_DIR, exist_ok=True)

    exo_env, healthy_env, healthy_policy, exo_policy = build_envs()
    healthy_angles = healthy_reference(healthy_env, healthy_policy)
    print(f"Healthy reference: reaches {healthy_angles[-1]:.3f} rad "
          f"(target {TARGET_ANGLE})")

    all_rows = []
    t200 = {"A": [], "B": []}
    for s in range(args.seed0, args.seed0 + args.seeds):
        for cond in ("A", "B"):
            rows, traj = run_condition(s, cond, exo_env, exo_policy,
                                       healthy_angles)
            all_rows.extend(rows)
            t200[cond].append(traj)
        print(f"seed {s}: done A+B")

    # --- write per-task CSV ---
    import csv
    csv_path = os.path.join(OUT_DIR, "carryover_log.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # --- save task-200 trajectories (ragged -> pad to STEPS_PER_TASK) ---
    def pad(lst):
        out = np.full((len(lst), STEPS_PER_TASK), np.nan)
        for i, a in enumerate(lst):
            out[i, :len(a)] = a
        return out
    np.savez(os.path.join(OUT_DIR, "carryover_task200.npz"),
             A=pad(t200["A"]), B=pad(t200["B"]),
             healthy=healthy_angles, target=TARGET_ANGLE)

    # --- Step 3 report ---
    import statistics as st
    rowsA = [r for r in all_rows if r["condition"] == "A"]
    rowsB = [r for r in all_rows if r["condition"] == "B"]

    def by_task(rows, task, key):
        return [r[key] for r in rows if r["task"] == task]

    def last50(rows, key):
        return [r[key] for r in rows if r["task"] > N_TASKS - 50]

    mfA = by_task(rowsA, N_TASKS, "mf_start")
    mfB = by_task(rowsB, N_TASKS, "mf_start")
    rA, rB = by_task(rowsA, N_TASKS, "pearson_r"), by_task(rowsB, N_TASKS, "pearson_r")
    faA, faB = by_task(rowsA, N_TASKS, "final_angle"), by_task(rowsB, N_TASKS, "final_angle")
    gA, gB = by_task(rowsA, N_TASKS, "goal"), by_task(rowsB, N_TASKS, "goal")

    def ms(x):
        x = [v for v in x if v == v]
        return (st.mean(x), st.pstdev(x)) if x else (float("nan"), float("nan"))

    lines = []
    lines.append("STEP 3 - CARRYOVER RESULTS (mean +/- std over seeds)\n")
    lines.append(f"seeds={args.seeds}  tasks={N_TASKS}  init_MF={INIT_MF}  "
                 f"force_scale={FORCE_SCALE}  slowdown={ACT_SLOWDOWN}\n")
    a_m, a_s = ms(mfA); b_m, b_s = ms(mfB)
    lines.append(f"Cumulative fatigue (avg MF) entering task {N_TASKS}:")
    lines.append(f"  A (no assist)      : {a_m:.4f} +/- {a_s:.4f}")
    lines.append(f"  B (assisted 1-199) : {b_m:.4f} +/- {b_s:.4f}")
    if a_m:
        lines.append(f"  reduction B vs A   : {100*(a_m-b_m)/a_m:+.1f}%")
    for name, A, B in [(f"Task-{N_TASKS} final angle (rad)", faA, faB),
                       (f"Task-{N_TASKS} Pearson r", rA, rB),
                       (f"Task-{N_TASKS} goal success", gA, gB)]:
        am, asd = ms(A); bm, bsd = ms(B)
        lines.append(f"{name}:  A={am:.3f}+/-{asd:.3f}   B={bm:.3f}+/-{bsd:.3f}")
    glA, glB = ms(last50(rowsA, "goal")), ms(last50(rowsB, "goal"))
    lines.append(f"Goal rate over last 50 tasks:  A={glA[0]:.3f}   B={glB[0]:.3f}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "carryover_summary.txt"), "w") as f:
        f.write(report + "\n")

    exo_env.close(); healthy_env.close()
    print(f"\nWrote {csv_path} and carryover_task200.npz")


if __name__ == "__main__":
    main()
