"""
Bayesian hyperparameter optimisation for the exo PPO policy.

Usage:
    python bayes_tune.py              # MLP tuning (brady+deg env)
    python bayes_tune.py --lstm        # CNN tuning — separate DB and output file
    python bayes_tune.py --reset      # delete study DB and start fresh
    python bayes_tune.py --lstm --reset

MLP output : best_params.json      (used by train_exo.py without --lstm)
CNN output : best_params_lstm.json  (used by train_exo.py with --lstm)

The CNN search space uses a lower learning-rate ceiling and larger n_steps
options since CNNs need smaller gradients and longer rollouts than MLPs.
"""

import argparse
import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import myosuite
from myosuite.utils import gym
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.temporal_buffer import TemporalStackWrapper
from models.cnn_lstm import CNNLSTMExtractor

HEALTHY_POLICY_PATH = "policies/healthy_policy.zip"
TUNE_TIMESTEPS   = 75_000
N_TRIALS         = 20
N_EVAL_EPISODES  = 10
PRUNE_EVAL_EP    = 5
PRUNE_EVAL_FREQ  = 10_000

# These are overridden in __main__ based on --lstm flag
_USE_CNN    = False
_STUDY_DB   = "sqlite:///bayes_study.db"
_STUDY_NAME = "exo_ppo_tuning"
_OUTPUT_PATH = "best_params.json"
_PLOT_DIR    = "plots/bayes"

PARAM_LABELS = {
    "learning_rate": "Learning Rate",
    "n_steps":       "N Steps",
    "batch_size":    "Batch Size",
    "ent_coef":      "Entropy Coeff",
    "gamma":         "Gamma",
    "clip_range":    "Clip Range",
}


# ── Environment ───────────────────────────────────────────────────────────────

def make_env():
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base,
        frozen_policy_path=HEALTHY_POLICY_PATH,
        bradykinesia=True,
        smart_reset=True,
        hide_pose_err=True,
    )
    if _USE_CNN:
        env = TemporalStackWrapper(env, window=20)
    return env


# ── Pruning callback ──────────────────────────────────────────────────────────

class PruningCallback(BaseCallback):
    """
    Evaluates the model every PRUNE_EVAL_FREQ steps, reports to Optuna,
    and raises TrialPruned when the pruner says to stop.
    """

    def __init__(self, trial: optuna.Trial, eval_env, eval_freq: int, n_eval_ep: int):
        super().__init__(verbose=0)
        self.trial     = trial
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.n_eval_ep = n_eval_ep
        self._checkpoint = 0

    def _on_step(self) -> bool:
        if self.num_timesteps >= (self._checkpoint + 1) * self.eval_freq:
            self._checkpoint += 1
            mean_reward, _ = evaluate_policy(
                self.model, self.eval_env,
                n_eval_episodes=self.n_eval_ep,
                deterministic=True,
                warn=False,
            )
            self.trial.report(mean_reward, self._checkpoint)
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return True


# ── Objective ─────────────────────────────────────────────────────────────────

def objective(trial: optuna.Trial) -> float:
    if _USE_CNN:
        # CNN needs lower LR and longer rollouts than MLP
        lr         = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
        n_steps    = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    else:
        lr         = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        n_steps    = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    ent_coef   = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    gamma      = trial.suggest_float("gamma", 0.95, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)

    policy_kwargs = None
    if _USE_CNN:
        policy_kwargs = dict(
            features_extractor_class=CNNLSTMExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        )

    env      = make_env()
    eval_env = make_env()
    try:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            ent_coef=ent_coef,
            gamma=gamma,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0,
        )
        callback = PruningCallback(trial, eval_env, PRUNE_EVAL_FREQ, PRUNE_EVAL_EP)
        model.learn(total_timesteps=TUNE_TIMESTEPS, callback=callback)
        mean_reward, _ = evaluate_policy(
            model, env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            warn=False,
        )
    except optuna.exceptions.TrialPruned:
        raise
    finally:
        env.close()
        eval_env.close()

    return float(mean_reward)


# ── Visualisation ─────────────────────────────────────────────────────────────

def _correlation_importance(study: optuna.Study) -> dict:
    """Absolute Pearson |r| between each hyperparameter and final reward."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed) < 3:
        return {}
    rewards = np.array([t.value for t in completed], dtype=float)
    importance = {}
    for param in PARAM_LABELS:
        vals = np.array([t.params.get(param, np.nan) for t in completed], dtype=float)
        mask = ~np.isnan(vals)
        if mask.sum() < 3:
            importance[param] = 0.0
            continue
        r = np.corrcoef(vals[mask], rewards[mask])[0, 1]
        importance[param] = 0.0 if np.isnan(r) else abs(r)
    return importance


def plot_study(study: optuna.Study) -> None:
    os.makedirs(PLOT_DIR, exist_ok=True)

    COMPLETE = optuna.trial.TrialState.COMPLETE
    PRUNED   = optuna.trial.TrialState.PRUNED

    all_trials = sorted(study.trials, key=lambda t: t.number)
    completed  = [t for t in all_trials if t.state == COMPLETE]
    pruned     = [t for t in all_trials if t.state == PRUNED]

    BLUE = "#4C72B0"
    RED  = "#DD4949"
    GREEN = "#2CA02C"

    # ── Plot 1: Trial overview ────────────────────────────────────────────────
    fig, (ax_bar, ax_best) = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle("Bayesian Optimisation — Trial Overview", fontsize=14, fontweight="bold")

    for t in completed:
        ax_bar.bar(t.number, t.value, color=BLUE, alpha=0.85, zorder=3)
    for t in pruned:
        last = max(t.intermediate_values.values()) if t.intermediate_values else 0.0
        ax_bar.bar(t.number, last, color=RED, alpha=0.6, hatch="//", zorder=3)

    ax_bar.set_xlabel("Trial Number")
    ax_bar.set_ylabel("Mean Reward")
    ax_bar.set_title("Reward per Trial")
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.85, label=f"Completed ({len(completed)})"),
        plt.Rectangle((0, 0), 1, 1, color=RED,  alpha=0.6,  hatch="//",
                       label=f"Pruned ({len(pruned)})"),
    ])

    current_best = -np.inf
    best_x, best_y = [], []
    for t in all_trials:
        if t.state == COMPLETE and t.value > current_best:
            current_best = t.value
        best_x.append(t.number)
        best_y.append(current_best if current_best > -np.inf else np.nan)

    ax_best.plot(best_x, best_y, color=GREEN, linewidth=2.5, marker="o", markersize=4)
    ax_best.set_xlabel("Trial Number")
    ax_best.set_ylabel("Best Reward So Far")
    ax_best.set_title("Running Best Reward")
    ax_best.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "trial_overview.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # ── Plot 2: Hyperparameter importance ────────────────────────────────────
    importance = _correlation_importance(study)
    if importance:
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        labels = [PARAM_LABELS[k] for k, _ in ranked]
        vals   = [v for _, v in ranked]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.Blues(np.linspace(0.35, 0.90, len(labels)))[::-1]
        bars = ax.barh(labels, vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=9)
        ax.set_xlabel("|Pearson r| with Final Reward")
        ax.set_xlim(0, 1.05)
        ax.set_title("Hyperparameter Importance\n(absolute correlation with final reward)",
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, "param_importance.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    # ── Plot 3: Hyperparameter scatter grid ──────────────────────────────────
    params_list = list(PARAM_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Hyperparameter vs Reward (completed=circles, pruned=×)",
                 fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    c_rewards = [t.value for t in completed]
    vmin = min(c_rewards) if c_rewards else 0
    vmax = max(c_rewards) if c_rewards else 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis
    last_sc = None

    for i, param in enumerate(params_list):
        ax = axes_flat[i]

        cx = [t.params[param] for t in completed if param in t.params]
        cy = [t.value         for t in completed if param in t.params]
        if cx:
            last_sc = ax.scatter(cx, cy, c=cy, cmap=cmap, norm=norm,
                                 s=65, alpha=0.85, zorder=4)

        px = [t.params[param] for t in pruned if param in t.params]
        py = [max(t.intermediate_values.values()) if t.intermediate_values else 0.0
              for t in pruned if param in t.params]
        if px:
            ax.scatter(px, py, marker="x", color=RED, s=65, linewidths=1.8,
                       zorder=5, label="pruned")

        ax.set_xlabel(PARAM_LABELS[param])
        ax.set_ylabel("Reward" if i % 3 == 0 else "")
        ax.grid(alpha=0.3)
        if param == "learning_rate":
            ax.set_xscale("log")

    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes.ravel().tolist(), label="Reward",
                     shrink=0.55, pad=0.02)
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "hyperparam_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    # ── Plot 4: Pruner timeline ───────────────────────────────────────────────
    trials_with_ints = [t for t in all_trials if t.intermediate_values]
    if trials_with_ints:
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.suptitle("Pruner Timeline — Intermediate Rewards per Trial",
                     fontsize=13, fontweight="bold")

        for t in trials_with_ints:
            checkpoints = sorted(t.intermediate_values.keys())
            steps = [c * PRUNE_EVAL_FREQ for c in checkpoints]
            vals  = [t.intermediate_values[c] for c in checkpoints]
            is_pruned = t.state == PRUNED
            color = RED  if is_pruned else BLUE
            alpha = 0.50 if is_pruned else 0.80
            lw    = 1.2  if is_pruned else 1.8
            ax.plot(steps, vals, color=color, linewidth=lw, alpha=alpha)
            if is_pruned:
                ax.scatter(steps[-1:], vals[-1:], color=RED, marker="x",
                           s=90, linewidths=2.2, zorder=5)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Intermediate Mean Reward (5-ep eval)")
        ax.grid(alpha=0.3)
        ax.legend(handles=[
            plt.Line2D([0], [0], color=BLUE, linewidth=2,   label=f"Completed ({len(completed)})"),
            plt.Line2D([0], [0], color=RED,  linewidth=1.5, label=f"Pruned ({len(pruned)}) — × marks cut point"),
        ])
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, "pruner_timeline.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")
    else:
        print("  (No intermediate values recorded — pruner timeline skipped)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter optimisation for exo PPO"
    )
    parser.add_argument("--lstm", action="store_true",
                        help="Tune for CNN policy (saves best_params_lstm.json)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete existing study DB and start fresh")
    args = parser.parse_args()

    # Set globals based on mode
    _USE_CNN = args.cnn  # noqa: F811
    if args.cnn:
        _STUDY_DB    = "sqlite:///bayes_study_lstm.db"   # noqa: F811
        _STUDY_NAME  = "exo_ppo_lstm_tuning"             # noqa: F811
        _OUTPUT_PATH = "best_params_lstm.json"           # noqa: F811
        _PLOT_DIR    = "plots/bayes_lstm"                # noqa: F811
        db_file      = "bayes_study_lstm.db"
    else:
        _STUDY_DB    = "sqlite:///bayes_study.db"       # noqa: F811
        _STUDY_NAME  = "exo_ppo_tuning"                 # noqa: F811
        _OUTPUT_PATH = "best_params.json"               # noqa: F811
        _PLOT_DIR    = "plots/bayes"                    # noqa: F811
        db_file      = "bayes_study.db"

    # Push into module-level names so make_env/objective see them
    import sys
    _mod = sys.modules[__name__]
    _mod._USE_CNN    = _USE_CNN
    _mod._STUDY_DB   = _STUDY_DB
    _mod._STUDY_NAME = _STUDY_NAME
    _mod._OUTPUT_PATH = _OUTPUT_PATH
    _mod._PLOT_DIR   = _PLOT_DIR

    if not os.path.exists(HEALTHY_POLICY_PATH):
        raise FileNotFoundError(
            f"Run train_healthy.py first — {HEALTHY_POLICY_PATH} not found."
        )

    if args.reset and os.path.exists(db_file):
        os.remove(db_file)
        print(f"Deleted {db_file} — starting fresh.")

    mode_label = "CNN" if args.cnn else "MLP"
    print("=" * 60)
    print(f"Bayesian optimisation [{mode_label}] — {N_TRIALS} trials x {TUNE_TIMESTEPS:,} steps")
    print(f"  bradykinesia=True  output={_OUTPUT_PATH}")
    print("=" * 60)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=_STUDY_DB,
        study_name=_STUDY_NAME,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nBest params: {best}")
    print(f"Best mean reward: {study.best_value:.4f}")

    with open(_OUTPUT_PATH, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved to {_OUTPUT_PATH}")

    print("\nGenerating visualisations...")
    plot_study(study)
    print(f"Plots saved to {_PLOT_DIR}/")
    print(f"\nNext step: python train_exo.py {'--lstm' if args.cnn else ''}")
