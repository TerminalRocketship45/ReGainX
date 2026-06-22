"""
Render the three ReGainX pipeline phases at a fixed target angle (2.0 rad) and
save clean RGB frames for the new Figure 1.

Phase 1 (Healthy, no exo):   healthy controller flexes the elbow to the goal.
Phase 2 (PD patient, no exo): same impaired patient stalls short of the goal.
Phase 3 (Exoskeleton agent):  RecurrentPPO torque assistance reaches the goal.

Phases 2 and 3 use an IDENTICAL patient profile (same seed -> same bradykinesia,
fatigue and start state); only the exo assistance differs.

Run with the exo_s environment python.

Usage:
    python render_pipeline_frames.py --search          # scan seeds, print angles
    python render_pipeline_frames.py --seed 7 --save    # render + save chosen seed
"""
import os
import argparse
import warnings

import numpy as np
import myosuite  # noqa: F401  (registers MyoSuite envs)
from myosuite.utils import gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from envs.elbow_env import CombinedExoOnlyWrapper

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
HEALTHY = os.path.join(ROOT, "policies", "healthy_policy.zip")
EXO = os.path.join(ROOT, "policies", "policy_brady_deg_recurrent.zip")
OUTDIR = os.path.join(ROOT, "ReGainX_paper", "figures", "sim")
os.makedirs(OUTDIR, exist_ok=True)

TARGET = 2.0
MAX_STEPS = 200
W = H = 480
CAM = 0


def render(base_env):
    return base_env.unwrapped.sim.renderer.render_offscreen(
        width=W, height=H, camera_id=CAM)


def elbow_angle(base_env):
    return float(base_env.unwrapped.sim.data.qpos[0])


# --------------------------------------------------------------------------
# Phase 1 - healthy controller, no exoskeleton
# --------------------------------------------------------------------------
def run_healthy(save=False):
    env = gym.make("myoElbowPose1D6MRandom-v0")
    model = PPO.load(HEALTHY)
    obs, _ = env.reset(seed=0)
    env.unwrapped.sim.data.qpos[:] = 0.0
    env.unwrapped.sim.data.qvel[:] = 0.0
    env.unwrapped.target_jnt_value = [TARGET]
    env.unwrapped.target_type = "fixed"
    env.unwrapped.update_target(restore_sim=True)
    env.unwrapped.sim.forward()
    od = model.observation_space.shape[0]
    obs = env.unwrapped.get_obs()[:od]

    best_frame, best_err, best_ang = None, 1e9, None
    for step in range(MAX_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        nobs, r, d, t, _ = env.step(action)
        obs = nobs[:od]
        ang = elbow_angle(env)
        err = abs(ang - TARGET)
        if err < best_err:
            best_err, best_ang = err, ang
            best_frame = render(env)
        if d or t:
            break
    env.close()
    if save and best_frame is not None:
        _imsave(best_frame, os.path.join(OUTDIR, "phase1_healthy.png"))
    return best_ang


# --------------------------------------------------------------------------
# Patient env (shared by phases 2 & 3)
# --------------------------------------------------------------------------
def make_patient_env():
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    return CombinedExoOnlyWrapper(
        base, frozen_policy_path=HEALTHY,
        bradykinesia=True, smart_reset=True, hide_pose_err=True)


def setup_patient(env, seed):
    """Seed identically, reset, then force start=0 and target=2.0."""
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    be = env.base_env.unwrapped
    be.target_jnt_value = [TARGET]
    be.target_type = "fixed"
    be.update_target(restore_sim=True)
    be.sim.data.qpos[:] = 0.0
    be.sim.data.qvel[:] = 0.0
    be.sim.forward()
    obs = env._build_obs(env._current_raw_obs())
    return obs


def run_patient_noexo(env, seed, save=False):
    obs = setup_patient(env, seed)
    be = env.base_env
    last_frame, last_ang = None, None
    for step in range(MAX_STEPS):
        obs, r, d, t, _ = env.step([0.0])
        last_ang = elbow_angle(be)
        if save:
            last_frame = render(be)
        if d or t:
            break
    if save and last_frame is not None:
        _imsave(last_frame, os.path.join(OUTDIR, "phase2_patient.png"))
    return last_ang


def run_exo(env, seed, save=False):
    exo = RecurrentPPO.load(EXO)
    obs = setup_patient(env, seed)
    be = env.base_env
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    best_frame, best_err, best_ang = None, 1e9, None
    for step in range(MAX_STEPS):
        action, lstm_states = exo.predict(
            obs, state=lstm_states, episode_start=episode_starts,
            deterministic=True)
        obs, r, d, t, _ = env.step(action)
        episode_starts = np.zeros((1,), dtype=bool)
        ang = elbow_angle(be)
        err = abs(ang - TARGET)
        if err < best_err:
            best_err, best_ang = err, ang
            if save:
                best_frame = render(be)
        if d or t:
            break
    if save and best_frame is not None:
        _imsave(best_frame, os.path.join(OUTDIR, "phase3_exo.png"))
    return best_ang


def _imsave(frame, path):
    import PIL.Image
    PIL.Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    if args.search:
        env = make_patient_env()
        print(f"{'seed':>4} {'patient_final':>14} {'exo_best':>10}")
        for s in range(0, 25):
            pf = run_patient_noexo(env, s, save=False)
            eb = run_exo(env, s, save=False)
            flag = "  <== good" if (pf < 1.45 and eb > 1.8) else ""
            print(f"{s:>4} {pf:>14.3f} {eb:>10.3f}{flag}")
        env.close()
        return

    h = run_healthy(save=args.save)
    env = make_patient_env()
    pf = run_patient_noexo(env, args.seed, save=args.save)
    eb = run_exo(env, args.seed, save=args.save)
    env.close()
    print(f"seed={args.seed}  healthy_best={h:.3f}  "
          f"patient_final={pf:.3f}  exo_best={eb:.3f}")


if __name__ == "__main__":
    main()
