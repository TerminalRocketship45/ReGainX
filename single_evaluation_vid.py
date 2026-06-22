"""
single_evaluation_vid.py  --  Record one episode of any policy.

Records side-by-side: chosen policy (left) vs healthy reference (right),
both using the same randomized start angle and target angle.
Prints start angle, target angle, and radian travelled to confirm randomization.

Usage
-----
  python single_evaluation_vid.py --policy policy_brady_deg_recurrent
  python single_evaluation_vid.py --policy policy_brady_deg
  python single_evaluation_vid.py --policy no_exo
  python single_evaluation_vid.py --policy policy_brady_deg_recurrent --seed 7

Supported policy names
  policy_brady_deg_recurrent
  policy_deg_recurrent
  policy_brady_deg
  policy_deg
  policy_brady_deg_recurrent_noisy
  no_exo

Output
------
  videos/single_eval/{policy}_vs_healthy.mp4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import myosuite  # noqa: F401
from myosuite.utils import gym
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECPPO = True
except ImportError:
    _HAS_RECPPO = False

from envs.elbow_env import CombinedExoOnlyWrapper
from utils import add_text_to_frame, save_video

ROOT         = Path(__file__).parent
POLICIES_DIR = ROOT / "policies"
HEALTHY_PATH = POLICIES_DIR / "healthy_policy.zip"
MAX_STEPS    = 200

RECURRENT_POLICIES = {
    "policy_brady_deg_recurrent",
    "policy_deg_recurrent",
    "policy_brady_deg_recurrent_noisy",
}


def make_exo_env():
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    return CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=str(HEALTHY_PATH),
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )


def load_exo_policy(name):
    path = POLICIES_DIR / f"{name}.zip"
    if not path.exists():
        sys.exit(f"Policy file not found: {path}")
    if name in RECURRENT_POLICIES:
        if not _HAS_RECPPO:
            sys.exit("sb3_contrib not installed — cannot load RecurrentPPO policy.")
        return RecurrentPPO.load(str(path)), True
    return PPO.load(str(path)), False


def main():
    parser = argparse.ArgumentParser(description="Record one episode of a policy")
    parser.add_argument("--policy", required=True,
                        help="Policy name, e.g. policy_brady_deg_recurrent")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (omit for a different episode each run)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    is_noexo = (args.policy == "no_exo")

    # ------------------------------------------------------------------
    # Build exo env; reset gives random start + target from MyoSuite
    # ------------------------------------------------------------------
    exo_env  = make_exo_env()
    base_env = exo_env

    exo_env.reset()
    start_angle   = float(base_env.base_env.unwrapped.sim.data.qpos[0])
    target_angle  = float(base_env.base_env.unwrapped.target_jnt_value[0])
    rad_travelled = abs(target_angle - start_angle)

    print(f"\n{'='*55}")
    print(f"  Policy        : {args.policy}")
    print(f"  Start angle   : {start_angle:.4f} rad  ({np.degrees(start_angle):.1f} deg)")
    print(f"  Target angle  : {target_angle:.4f} rad  ({np.degrees(target_angle):.1f} deg)")
    print(f"  Rad travelled : {rad_travelled:.4f} rad  ({np.degrees(rad_travelled):.1f} deg)")
    print(f"{'='*55}\n")

    # Rebuild obs from the reset state (qpos is already at start_angle)
    exo_obs = base_env._build_obs(base_env._current_raw_obs())

    # ------------------------------------------------------------------
    # Load policies
    # ------------------------------------------------------------------
    exo_policy, is_recurrent = (None, False) if is_noexo else load_exo_policy(args.policy)
    healthy_policy = PPO.load(str(HEALTHY_PATH))
    obs_dim        = healthy_policy.observation_space.shape[0]

    # ------------------------------------------------------------------
    # Set healthy env to the same start angle and target
    # ------------------------------------------------------------------
    healthy_env = gym.make("myoElbowPose1D6MRandom-v0")
    healthy_env.reset()
    healthy_env.unwrapped.target_jnt_value = [target_angle]
    healthy_env.unwrapped.target_type      = "fixed"
    healthy_env.unwrapped.update_target(restore_sim=True)
    healthy_env.unwrapped.sim.data.qpos[0] = start_angle
    healthy_env.unwrapped.sim.data.qvel[:] = 0.0
    healthy_env.unwrapped.sim.forward()
    h_obs = healthy_env.unwrapped.get_obs()[:obs_dim]

    # ------------------------------------------------------------------
    # Step both envs and record side-by-side frames
    # ------------------------------------------------------------------
    frames      = []
    exo_reward  = 0.0
    h_reward    = 0.0
    lstm_states = None
    ep_start    = np.ones((1,), dtype=bool)
    zero        = np.zeros(exo_env.action_space.shape, dtype=np.float32)
    exo_done    = False
    h_done      = False

    for step in range(MAX_STEPS):
        t = step * base_env.base_env.unwrapped.dt

        exo_label = (
            f"{args.policy}  "
            f"start={start_angle:.2f}rad  tgt={target_angle:.2f}rad  "
            f"t={t:.1f}s  cumR={exo_reward:.0f}"
        )
        h_label = (
            f"healthy_reference  "
            f"start={start_angle:.2f}rad  tgt={target_angle:.2f}rad  "
            f"t={t:.1f}s  cumR={h_reward:.0f}"
        )

        exo_frame = base_env.base_env.unwrapped.sim.renderer.render_offscreen(
            width=400, height=400, camera_id=0)
        h_frame   = healthy_env.unwrapped.sim.renderer.render_offscreen(
            width=400, height=400, camera_id=0)

        exo_frame = add_text_to_frame(np.asarray(exo_frame, dtype=np.uint8), exo_label)
        h_frame   = add_text_to_frame(np.asarray(h_frame,   dtype=np.uint8), h_label)

        frames.append(np.concatenate([exo_frame, h_frame], axis=1))

        # Step exo
        if not exo_done:
            if is_noexo:
                action = zero
            elif is_recurrent:
                action, lstm_states = exo_policy.predict(
                    exo_obs, state=lstm_states,
                    episode_start=ep_start, deterministic=True)
                ep_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = exo_policy.predict(exo_obs, deterministic=True)

            exo_obs, rwd, done, truncated, _ = exo_env.step(action)
            exo_reward += float(rwd)
            exo_done    = done or truncated

        # Step healthy
        if not h_done:
            h_action, _ = healthy_policy.predict(h_obs, deterministic=True)
            h_obs, h_rwd, h_d, h_t, _ = healthy_env.step(h_action)
            h_obs     = h_obs[:obs_dim]
            h_reward += float(h_rwd)
            h_done    = h_d or h_t

        if exo_done and h_done:
            break

    exo_env.close()
    healthy_env.close()

    out_path = ROOT / "videos" / "single_eval" / f"{args.policy}_vs_healthy.mp4"
    save_video(frames, str(out_path))

    print(f"  Exo reward     : {exo_reward:.2f}")
    print(f"  Healthy reward : {h_reward:.2f}")
    print(f"  Frames         : {len(frames)}")
    print(f"\n  Video saved -> {out_path}")


if __name__ == "__main__":
    main()
