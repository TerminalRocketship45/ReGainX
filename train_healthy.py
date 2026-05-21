"""
Train the frozen healthy elbow policy and record demo videos.
Run once before any other training script.

Usage:
    python train_healthy.py
"""

import os
import numpy as np
import myosuite  # registers MyoSuite envs
from myosuite.utils import gym
from stable_baselines3 import PPO
from utils import save_video, add_text_to_frame

POLICY_PATH = "policies/healthy_policy"
VIDEO_DIR = "videos/healthy_elbow"
TIMESTEPS = 250_000
DEMO_ANGLES = [0.5, 1.0, 1.5, 2.0, 2.5]  # rad
MAX_STEPS = 500


def train_healthy_policy() -> PPO:
    print("=" * 60)
    print("Training healthy elbow policy")
    print(f"  Env: myoElbowPose1D6MRandom-v0")
    print(f"  Timesteps: {TIMESTEPS:,}")
    print("=" * 60)

    env = gym.make("myoElbowPose1D6MRandom-v0")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=TIMESTEPS)
    os.makedirs("policies", exist_ok=True)
    model.save(POLICY_PATH)
    print(f"\nPolicy saved to {POLICY_PATH}.zip")
    env.close()
    return model


def record_demo_videos(model: PPO) -> None:
    os.makedirs(VIDEO_DIR, exist_ok=True)
    env = gym.make("myoElbowPose1D6MRandom-v0")

    for target_angle in DEMO_ANGLES:
        frames = []
        obs, _ = env.reset()

        # Set fixed start and target
        env.unwrapped.sim.data.qpos[:] = 0.0
        env.unwrapped.sim.data.qvel[:] = 0.0
        env.unwrapped.target_jnt_value = [target_angle]
        env.unwrapped.target_type = "fixed"
        env.unwrapped.update_target(restore_sim=True)
        env.unwrapped.sim.forward()

        # Re-fetch obs after state change
        obs = env.unwrapped.get_obs()[: model.observation_space.shape[0]]

        total_reward = 0.0
        for step in range(MAX_STEPS):
            frame = env.unwrapped.sim.renderer.render_offscreen(
                width=400, height=400, camera_id=0
            )
            t = step * env.unwrapped.dt
            label = f"Healthy  target={target_angle:.1f}rad  t={t:.1f}s"
            frames.append(add_text_to_frame(frame, label))

            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            obs = next_obs[: model.observation_space.shape[0]]
            total_reward += reward
            if done or truncated:
                break

        video_path = os.path.join(VIDEO_DIR, f"healthy_angle_{target_angle:.1f}.mp4")
        save_video(frames, video_path)
        print(f"  Saved {video_path}  (reward={total_reward:.1f}, {len(frames)} frames)")

    env.close()


if __name__ == "__main__":
    model = train_healthy_policy()
    print("\nRecording demo videos...")
    record_demo_videos(model)
    print("\nDone. Next step: run bayes_tune.py")
