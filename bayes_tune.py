"""
Bayesian hyperparameter optimisation for the exo PPO policy.
Run ONCE after train_healthy.py. Saves best_params.json for all train_exo.py runs.

Usage:
    python bayes_tune.py

Requires: policies/healthy_policy.zip (from train_healthy.py)
Output:   best_params.json, bayes_study.db
"""

import json
import os
import myosuite
from myosuite.utils import gym
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from envs.elbow_env import CombinedExoOnlyWrapper

HEALTHY_POLICY_PATH = "policies/healthy_policy.zip"
TUNE_TIMESTEPS = 75_000
N_TRIALS = 20
N_EVAL_EPISODES = 10
STUDY_DB = "sqlite:///bayes_study.db"
STUDY_NAME = "exo_ppo_tuning"
OUTPUT_PATH = "best_params.json"

SB3_DEFAULTS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "ent_coef": 0.0,
    "gamma": 0.99,
    "clip_range": 0.2,
}


def make_env():
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    return CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=False,
        smart_reset=True,
        hide_pose_err=True,
    )


def objective(trial: optuna.Trial) -> float:
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    env = make_env()
    try:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=gamma,
            clip_range=clip_range,
            verbose=0,
        )
        model.learn(total_timesteps=TUNE_TIMESTEPS)
        mean_reward, _ = evaluate_policy(
            model, env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
    finally:
        env.close()

    return float(mean_reward)


if __name__ == "__main__":
    if not os.path.exists(HEALTHY_POLICY_PATH):
        raise FileNotFoundError(
            f"Run train_healthy.py first — {HEALTHY_POLICY_PATH} not found."
        )

    print("=" * 60)
    print(f"Bayesian hyperparameter optimisation — {N_TRIALS} trials x {TUNE_TIMESTEPS:,} steps")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
        storage=STUDY_DB,
        study_name=STUDY_NAME,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest params: {best}")
    print(f"Best mean reward: {study.best_value:.4f}")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved to {OUTPUT_PATH}")
    print("\nNext step: run train_exo.py")
