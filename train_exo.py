"""
Train an exoskeleton policy with configurable impairment, CNN-LSTM, and extra-obs settings.

Bradykinesia is ON by default. Use --no-bradykinesia to train on fatigue only.
--extraobs appends avg_mf, force_scale, and activation_slowdown to the observation.
--lstm enables the CNN-LSTM feature extractor (2M steps recommended).

Usage:
    python train_exo.py                          # brady + deg, MLP
    python train_exo.py --no-bradykinesia        # deg only, MLP
    python train_exo.py --lstm                   # brady + deg + CNN-LSTM
    python train_exo.py --no-bradykinesia --lstm # deg only + CNN-LSTM
    python train_exo.py --lstm --extraobs        # brady + deg + CNN-LSTM + extra obs
    python train_exo.py --timesteps 2000000      # custom timestep count

Requires:
    policies/healthy_policy.zip   (from train_healthy.py)
    best_params.json              (from bayes_tune.py — falls back to SB3 defaults)
    best_params_lstm.json         (from bayes_tune.py --lstm — falls back to literature values)

Policy naming:
    policy_brady_deg.zip
    policy_deg.zip
    policy_brady_deg_lstm.zip
    policy_deg_lstm.zip
    policy_brady_deg_lstm_extraobs.zip
"""

import argparse
import json
import os
import time

import numpy as np
import myosuite
from myosuite.utils import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.temporal_buffer import TemporalStackWrapper
from models.cnn_lstm import CNNLSTMExtractor
from utils import save_video, add_text_to_frame

HEALTHY_POLICY_PATH = "policies/healthy_policy.zip"
PARAMS_PATH         = "best_params.json"
PARAMS_PATH_LSTM    = "best_params_lstm.json"
POST_ANALYSIS_EPISODES = 30
MAX_STEPS = 500

SB3_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.0,
    "gamma": 0.99,
    "clip_range": 0.2,
}

# Literature values for CNN-LSTM + PPO (Han et al. 2024; SB3 rl-baselines3-zoo)
LSTM_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.0,
    "gamma": 0.99,
    "clip_range": 0.2,
}


# ── Reward CSV logger ──────────────────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def policy_name(bradykinesia: bool, lstm: bool, extraobs: bool = False) -> str:
    brady        = "brady_deg" if bradykinesia else "deg"
    lstm_suffix  = "_lstm"     if lstm         else ""
    extra_suffix = "_extraobs" if extraobs      else ""
    return f"policy_{brady}{lstm_suffix}{extra_suffix}"


def load_params(lstm: bool = False) -> dict:
    path     = PARAMS_PATH_LSTM if lstm else PARAMS_PATH
    defaults = LSTM_DEFAULTS    if lstm else SB3_DEFAULTS
    fallback = PARAMS_PATH

    if os.path.exists(path):
        with open(path) as f:
            params = json.load(f)
        print(f"Loaded hyperparameters from {path}")
        return params
    if lstm and os.path.exists(fallback):
        with open(fallback) as f:
            params = json.load(f)
        print(f"WARNING: {path} not found — using {fallback}. Run: python bayes_tune.py --lstm")
        return params
    flag = " --lstm" if lstm else ""
    print(f"WARNING: {path} not found — using {'literature' if lstm else 'SB3'} defaults."
          f" Run: python bayes_tune.py{flag}")
    return defaults.copy()


def build_env(bradykinesia: bool, lstm: bool, extraobs: bool = False):
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=bradykinesia,
        smart_reset=True,
        hide_pose_err=True,
        extra_obs=extraobs,
    )
    if lstm:
        env = TemporalStackWrapper(env, window=20)
    return env


def build_model(env, params: dict, lstm: bool, log_dir: str) -> PPO:
    policy_kwargs = None
    if lstm:
        policy_kwargs = dict(
            features_extractor_class=CNNLSTMExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )

    return PPO(
        "MlpPolicy", env,
        learning_rate=params.get("learning_rate", SB3_DEFAULTS["learning_rate"]),
        n_steps=params.get("n_steps",      SB3_DEFAULTS["n_steps"]),
        batch_size=params.get("batch_size", SB3_DEFAULTS["batch_size"]),
        ent_coef=params.get("ent_coef",    SB3_DEFAULTS["ent_coef"]),
        gamma=params.get("gamma",          SB3_DEFAULTS["gamma"]),
        clip_range=params.get("clip_range", SB3_DEFAULTS["clip_range"]),
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        verbose=1,
    )


def post_training_analysis(model: PPO, env, pname: str, is_lstm: bool) -> None:
    """30-episode analysis: reward, goal rate, episode videos."""
    video_dir = f"videos/{pname}"
    os.makedirs(video_dir, exist_ok=True)

    rewards, goals, lengths = [], [], []

    for ep in range(POST_ANALYSIS_EPISODES):
        obs, _ = env.reset()
        total_reward = 0.0
        frames = []
        solved = False

        for step in range(MAX_STEPS):
            base_env = env.env if is_lstm else env
            frame = base_env.base_env.unwrapped.sim.renderer.render_offscreen(
                width=400, height=400, camera_id=0
            )
            t = step * base_env.base_env.unwrapped.dt
            label = f"{pname}  ep={ep+1}  t={t:.1f}s"
            frames.append(add_text_to_frame(frame, label))

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

            rwd_dict = base_env.base_env.unwrapped.rwd_dict
            solved_val = rwd_dict.get("solved", False)
            if bool(np.asarray(solved_val).flat[0]):
                solved = True
            if done or truncated:
                break

        rewards.append(total_reward)
        goals.append(solved)
        lengths.append(step + 1)

        path = os.path.join(video_dir, f"episode_{ep+1:03d}.mp4")
        save_video(frames, path)
        print(f"  Ep {ep+1:2d}/{POST_ANALYSIS_EPISODES}: reward={total_reward:7.2f}  "
              f"goal={'Y' if solved else 'N'}  frames={len(frames)}")

    print(f"\n{'='*60}")
    print(f"Post-training summary — {pname}")
    print(f"  Mean reward : {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Goal rate   : {sum(goals)}/{POST_ANALYSIS_EPISODES} "
          f"({100*np.mean(goals):.1f}%)")
    print(f"  Mean length : {np.mean(lengths):.1f} steps")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train exo PPO policy with optional bradykinesia and CNN-LSTM"
    )
    parser.add_argument("--no-bradykinesia", dest="bradykinesia",
                        action="store_false",
                        help="Disable bradykinesia (train on fatigue/degeneration only)")
    parser.set_defaults(bradykinesia=True)
    parser.add_argument("--lstm", action="store_true",
                        help="Use CNN-LSTM feature extractor (2M steps recommended)")
    parser.add_argument("--extraobs", action="store_true",
                        help="Append avg_mf, force_scale, activation_slowdown to observation")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps (default: 1,000,000)")
    args = parser.parse_args()

    pname  = policy_name(args.bradykinesia, args.lstm, args.extraobs)
    params = load_params(lstm=args.lstm)

    print("=" * 60)
    print(f"Training: {pname}")
    print(f"  bradykinesia={args.bradykinesia}  lstm={args.lstm}  extraobs={args.extraobs}")
    print(f"  timesteps={args.timesteps:,}")
    print(f"  lr={params.get('learning_rate'):.2e}  n_steps={params.get('n_steps')}  "
          f"batch_size={params.get('batch_size')}")
    print("=" * 60)

    env     = build_env(args.bradykinesia, args.lstm, args.extraobs)
    log_dir = f"tensorboard/{pname}/"
    model   = build_model(env, params, args.lstm, log_dir)

    os.makedirs("logs", exist_ok=True)
    reward_logger = RewardLoggerCallback(f"logs/{pname}_rewards.csv")

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=True,
        callback=reward_logger,
    )

    os.makedirs("policies", exist_ok=True)
    save_path = f"policies/{pname}"
    model.save(save_path)
    print(f"\nPolicy saved to {save_path}.zip")
    print(f"Reward log  → logs/{pname}_rewards.csv")

    print("\nRunning post-training analysis (30 episodes)...")
    post_training_analysis(model, env, pname, args.lstm)
    env.close()


if __name__ == "__main__":
    main()
