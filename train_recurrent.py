"""
Train a RecurrentPPO (PPO-LSTM) exoskeleton policy via sb3_contrib.

Bradykinesia is ON by default. Use --no-bradykinesia to train on fatigue/degeneration only.
RecurrentPPO carries its own internal LSTM so no TemporalStackWrapper is needed.

Usage:
    python train_recurrent.py                        # brady+deg, 2M steps
    python train_recurrent.py --no-bradykinesia      # deg only, 1M steps default
    python train_recurrent.py --timesteps 1000000

Policy naming:
    policy_brady_deg_recurrent.zip   (bradykinesia ON)
    policy_deg_recurrent.zip         (bradykinesia OFF)
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
    """Appends (timestep, mean_episode_reward) to a CSV whenever episodes complete.

    RecurrentPPO uses n_steps=256 which is shorter than a typical episode, so
    ep_info_buffer is empty at most rollout boundaries.  We track episode totals
    manually in _on_step instead and flush every LOG_EVERY completed episodes.
    """

    LOG_EVERY = 10  # write one CSV row per this many completed episodes

    def __init__(self, csv_path: str):
        super().__init__()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._path = csv_path
        self._current_reward = 0.0
        self._completed: list = []
        with open(csv_path, "w") as f:
            f.write("timestep,mean_reward\n")

    def _on_step(self) -> bool:
        self._current_reward += float(np.asarray(self.locals["rewards"]).flat[0])
        if bool(np.asarray(self.locals["dones"]).flat[0]):
            self._completed.append(self._current_reward)
            self._current_reward = 0.0
            if len(self._completed) >= self.LOG_EVERY:
                mean_r = float(np.mean(self._completed))
                self._completed.clear()
                with open(self._path, "a") as f:
                    f.write(f"{self.num_timesteps},{mean_r:.4f}\n")
        return True

    def _on_rollout_end(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Train RecurrentPPO (PPO-LSTM) exo policy"
    )
    parser.add_argument("--no-bradykinesia", dest="bradykinesia",
                        action="store_false",
                        help="Disable bradykinesia (train on fatigue/degeneration only)")
    parser.set_defaults(bradykinesia=True)
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default: 2M with brady, 1M without)")
    args = parser.parse_args()

    policy_name = "policy_brady_deg_recurrent" if args.bradykinesia else "policy_deg_recurrent"
    policy_path = f"policies/{policy_name}.zip"
    log_path    = f"logs/{policy_name}_rewards.csv"
    default_ts  = 2_000_000 if args.bradykinesia else 1_000_000
    timesteps   = args.timesteps if args.timesteps is not None else default_ts

    import sys
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_recurrent] Python:  {sys.executable}")
    print(f"[train_recurrent] PyTorch: {torch.__version__}")
    print(f"[train_recurrent] CUDA available: {torch.cuda.is_available()}")
    print(f"[train_recurrent] Using device: {device}")

    print("=" * 60)
    print(f"Training: {policy_name}")
    print(f"  algorithm=RecurrentPPO  bradykinesia={args.bradykinesia}  extraobs=False")
    print(f"  timesteps={timesteps:,}")
    print(f"  n_steps={RECURRENT_DEFAULTS['n_steps']}  "
          f"batch_size={RECURRENT_DEFAULTS['batch_size']}  "
          f"lr={RECURRENT_DEFAULTS['learning_rate']:.2e}")
    print("=" * 60)

    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=args.bradykinesia,
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
        tensorboard_log=f"tensorboard/{policy_name}/",
        verbose=1,
        device=device,
    )

    os.makedirs("logs", exist_ok=True)
    callback = RewardLoggerCallback(log_path)

    model.learn(
        total_timesteps=timesteps,
        reset_num_timesteps=True,
        callback=callback,
    )

    os.makedirs("policies", exist_ok=True)
    model.save(f"policies/{policy_name}")
    print(f"\nPolicy saved to {policy_path}")
    print(f"Reward log  → {log_path}")
    env.close()


if __name__ == "__main__":
    main()
