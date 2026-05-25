"""
Train RecurrentPPO (PPO-LSTM) with randomised EMG noise.

At the start of each episode, noise sigma is re-sampled from
Uniform(0.01, 0.10), covering SNR ≈ 14–34 dB — the realistic range for
surface EMG.  This domain-randomises the noise level so the recurrent
policy learns to be robust to measurement uncertainty.

Usage:
    python train_recurrent_noisy.py                    # brady+deg+noise, 1.5M steps
    python train_recurrent_noisy.py --no-bradykinesia  # deg+noise only
    python train_recurrent_noisy.py --timesteps 2000000

Output:
    policies/policy_brady_deg_recurrent_noisy.zip
    logs/policy_brady_deg_recurrent_noisy_rewards.csv
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

HEALTHY_POLICY_PATH = "policies/healthy_policy.zip"

# Realistic surface EMG noise range
#   sigma = 0.01  →  SNR ≈ 34 dB  (high quality recording)
#   sigma = 0.10  →  SNR ≈ 14 dB  (noisy / clinical setting)
NOISE_SIGMA_LOW  = 0.01
NOISE_SIGMA_HIGH = 0.10

RECURRENT_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps":       256,
    "batch_size":    64,
    "ent_coef":      0.0,
    "gamma":         0.99,
    "clip_range":    0.2,
}


class RewardLoggerCallback(BaseCallback):
    """Logs mean episode reward to CSV every LOG_EVERY completed episodes."""

    LOG_EVERY = 10

    def __init__(self, csv_path: str):
        super().__init__()
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        self._path            = csv_path
        self._current_reward  = 0.0
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
        description="Train RecurrentPPO with randomised EMG noise"
    )
    parser.add_argument(
        "--no-bradykinesia", dest="bradykinesia", action="store_false",
        help="Disable bradykinesia (train on fatigue/degeneration + noise only)",
    )
    parser.set_defaults(bradykinesia=True)
    parser.add_argument(
        "--timesteps", type=int, default=1_500_000,
        help="Total training timesteps (default: 1.5M)",
    )
    args = parser.parse_args()

    suffix      = "brady_deg" if args.bradykinesia else "deg"
    policy_name = f"policy_{suffix}_recurrent_noisy"
    policy_path = f"policies/{policy_name}.zip"
    log_path    = f"logs/{policy_name}_rewards.csv"

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    snr_low  = 20 * np.log10(0.5 / NOISE_SIGMA_HIGH)
    snr_high = 20 * np.log10(0.5 / NOISE_SIGMA_LOW)

    print(f"[train_recurrent_noisy] Python:  {sys.executable}")
    print(f"[train_recurrent_noisy] PyTorch: {torch.__version__}")
    print(f"[train_recurrent_noisy] Using device: {device}")
    print("=" * 60)
    print(f"Training: {policy_name}")
    print(f"  algorithm    = RecurrentPPO (PPO-LSTM)")
    print(f"  bradykinesia = {args.bradykinesia}")
    print(f"  EMG noise    = Uniform({NOISE_SIGMA_LOW}, {NOISE_SIGMA_HIGH}) per episode")
    print(f"               = SNR ≈ {snr_low:.0f} – {snr_high:.0f} dB (normalised signal)")
    print(f"  timesteps    = {args.timesteps:,}")
    print(f"  n_steps      = {RECURRENT_DEFAULTS['n_steps']}")
    print(f"  batch_size   = {RECURRENT_DEFAULTS['batch_size']}")
    print(f"  lr           = {RECURRENT_DEFAULTS['learning_rate']:.2e}")
    print("=" * 60)

    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    inner = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=args.bradykinesia,
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
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
        callback=callback,
    )

    os.makedirs("policies", exist_ok=True)
    model.save(f"policies/{policy_name}")
    print(f"\nPolicy saved → {policy_path}")
    print(f"Reward log   → {log_path}")
    env.close()


if __name__ == "__main__":
    main()
