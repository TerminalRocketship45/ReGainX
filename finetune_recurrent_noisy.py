"""
Continue training policy_brady_deg_recurrent_noisy for 1.5M more timesteps
under the simplified white-noise NoisyExoWrapper.

Loads the existing policy, attaches the current noise env, trains, and saves
back to the same path — overwriting the old weights.

Usage:
    python finetune_recurrent_noisy.py
    python finetune_recurrent_noisy.py --timesteps 2000000

Output (overwrites existing):
    policies/policy_brady_deg_recurrent_noisy.zip
    logs/policy_brady_deg_recurrent_noisy_finetune_rewards.csv
"""

import argparse
import os
import sys

import numpy as np
import myosuite  # noqa: F401
from myosuite.utils import gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.elbow_env_noisy import NoisyExoWrapper

POLICY_PATH         = "policies/policy_brady_deg_recurrent_noisy.zip"
HEALTHY_POLICY_PATH = "policies/healthy_policy.zip"
LOG_PATH            = "logs/policy_brady_deg_recurrent_noisy_finetune_rewards.csv"

NOISE_SIGMA_LOW  = 0.01
NOISE_SIGMA_HIGH = 0.10

RECURRENT_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps":       256,
    "batch_size":    64,
}


class RewardLoggerCallback(BaseCallback):
    """Logs mean episode reward to CSV every LOG_EVERY completed episodes."""

    LOG_EVERY = 10

    def __init__(self, csv_path: str):
        super().__init__()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._path           = csv_path
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
        description="Continue training the noisy recurrent policy for more timesteps"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1_500_000,
        help="Additional training timesteps (default: 1.5M)",
    )
    args = parser.parse_args()

    if not os.path.exists(POLICY_PATH):
        print(f"ERROR: policy not found at {POLICY_PATH}")
        print("  Run train_recurrent_noisy.py first to create the initial policy.")
        sys.exit(1)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    snr_low  = 20 * np.log10(0.5 / NOISE_SIGMA_HIGH)
    snr_high = 20 * np.log10(0.5 / NOISE_SIGMA_LOW)

    print(f"[finetune_recurrent_noisy] Python:  {sys.executable}")
    print(f"[finetune_recurrent_noisy] PyTorch: {torch.__version__}")
    print(f"[finetune_recurrent_noisy] Device:  {device}")
    print("=" * 60)
    print(f"Continuing: {POLICY_PATH}")
    print(f"  additional timesteps = {args.timesteps:,}")
    print(f"  EMG noise            = Uniform({NOISE_SIGMA_LOW}, {NOISE_SIGMA_HIGH}) per episode")
    print(f"                       = SNR ≈ {snr_low:.0f} – {snr_high:.0f} dB")
    print(f"  noise model          = white noise (zero-mean N(0,sigma) per step)")
    print(f"  output               = {POLICY_PATH}  [overwrites]")
    print("=" * 60)

    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    inner = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=False,
    )
    env = NoisyExoWrapper(
        inner,
        randomize_sigma=True,
        sigma_low=NOISE_SIGMA_LOW,
        sigma_high=NOISE_SIGMA_HIGH,
    )

    print(f"Loading policy from {POLICY_PATH} ...")
    model = RecurrentPPO.load(POLICY_PATH, env=env, device=device)

    # Keep the same hyperparameters as the original training run
    model.learning_rate = RECURRENT_DEFAULTS["learning_rate"]
    model.n_steps       = RECURRENT_DEFAULTS["n_steps"]
    model.batch_size    = RECURRENT_DEFAULTS["batch_size"]

    os.makedirs("logs", exist_ok=True)
    callback = RewardLoggerCallback(LOG_PATH)

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
        callback=callback,
    )

    os.makedirs("policies", exist_ok=True)
    model.save("policies/policy_brady_deg_recurrent_noisy")
    print(f"\nPolicy saved  → {POLICY_PATH}")
    print(f"Reward log    → {LOG_PATH}")
    env.close()


if __name__ == "__main__":
    main()
