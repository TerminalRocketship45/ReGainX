"""
Train a RecurrentPPO (PPO-LSTM) exoskeleton policy via sb3_contrib.

Trains on the brady+deg environment (bradykinesia ON, no extra obs) for
2,000,000 timesteps. RecurrentPPO carries its own internal LSTM so no
TemporalStackWrapper is needed.

Usage:
    python train_recurrent.py
    python train_recurrent.py --timesteps 2000000

Policy saved to:
    policies/policy_brady_deg_recurrent.zip
    logs/policy_brady_deg_recurrent_rewards.csv
"""

import argparse
import os

import numpy as np
import myosuite
from myosuite.utils import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.elbow_env import CombinedExoOnlyWrapper

HEALTHY_POLICY_PATH = "policies/healthy_policy.zip"
POLICY_NAME = "policy_brady_deg_recurrent"
POLICY_PATH = f"policies/{POLICY_NAME}.zip"
LOG_PATH = f"logs/{POLICY_NAME}_rewards.csv"

# Tuned for RecurrentPPO: shorter rollouts so LSTM state resets more frequently.
# n_steps * n_envs must be divisible by batch_size.
RECURRENT_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 256,
    "batch_size": 64,
    "ent_coef": 0.0,
    "gamma": 0.99,
    "clip_range": 0.2,
}


class RewardLoggerCallback(BaseCallback):
    """Appends (timestep, mean_episode_reward) to a CSV at each rollout end."""

    def __init__(self, csv_path: str):
        super().__init__()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._path = csv_path
        with open(csv_path, "w") as f:
            f.write("timestep,mean_reward\n")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_r = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            with open(self._path, "a") as f:
                f.write(f"{self.num_timesteps},{mean_r:.4f}\n")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Train RecurrentPPO (PPO-LSTM) on brady+deg exo environment"
    )
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total training timesteps (default: 2,000,000)")
    args = parser.parse_args()

    import sys
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_recurrent] Python:  {sys.executable}")
    print(f"[train_recurrent] PyTorch: {torch.__version__}")
    print(f"[train_recurrent] CUDA available: {torch.cuda.is_available()}")
    print(f"[train_recurrent] Using device: {device}")

    print("=" * 60)
    print(f"Training: {POLICY_NAME}")
    print(f"  algorithm=RecurrentPPO  bradykinesia=True  extraobs=False")
    print(f"  timesteps={args.timesteps:,}")
    print(f"  n_steps={RECURRENT_DEFAULTS['n_steps']}  "
          f"batch_size={RECURRENT_DEFAULTS['batch_size']}  "
          f"lr={RECURRENT_DEFAULTS['learning_rate']:.2e}")
    print("=" * 60)

    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=RECURRENT_DEFAULTS["learning_rate"],
        n_steps=RECURRENT_DEFAULTS["n_steps"],
        batch_size=RECURRENT_DEFAULTS["batch_size"],
        ent_coef=RECURRENT_DEFAULTS["ent_coef"],
        gamma=RECURRENT_DEFAULTS["gamma"],
        clip_range=RECURRENT_DEFAULTS["clip_range"],
        tensorboard_log=f"tensorboard/{POLICY_NAME}/",
        verbose=1,
        device=device,
    )

    os.makedirs("logs", exist_ok=True)
    callback = RewardLoggerCallback(LOG_PATH)

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
        callback=callback,
    )

    os.makedirs("policies", exist_ok=True)
    model.save(f"policies/{POLICY_NAME}")
    print(f"\nPolicy saved to {POLICY_PATH}")
    print(f"Reward log  → {LOG_PATH}")
    env.close()


if __name__ == "__main__":
    main()
