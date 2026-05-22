"""
Full reGainX pipeline: train → chart → evaluate → compare.

Trains all missing policies with the appropriate timestep budget, then
generates a combined reward-curve chart, runs per-policy evaluations, and
runs all three ablation comparisons via compare.py.

Timestep budget:
    MLP  policies (policy_brady_deg, policy_deg)            → 1,000,000 steps
    LSTM policies (policy_brady_deg_lstm, policy_deg_lstm,
                   policy_brady_deg_lstm_extraobs)           → 2,000,000 steps

Usage:
    python pipeline.py

Requires:
    policies/healthy_policy.zip   (from train_healthy.py)

Trains any of the following that are missing:
    policies/policy_brady_deg.zip
    policies/policy_deg.zip
    policies/policy_brady_deg_lstm.zip
    policies/policy_deg_lstm.zip
    policies/policy_brady_deg_lstm_extraobs.zip

Outputs:
    results/pipeline/reward_curves_all_models.png
    results/eval_policy_*/
    results/lstm_ablation/
    results/bradykinesia_ablation/
    results/extraobs_ablation/
"""

import os
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PYTHON = sys.executable

HEALTHY_PATH    = "policies/healthy_policy.zip"
POLICY_BRADY    = "policies/policy_brady_deg.zip"
POLICY_DEG      = "policies/policy_deg.zip"
POLICY_LSTM     = "policies/policy_brady_deg_lstm.zip"
POLICY_DEG_LSTM = "policies/policy_deg_lstm.zip"
POLICY_EXTRAOBS = "policies/policy_brady_deg_lstm_extraobs.zip"

LOG_BRADY    = "logs/policy_brady_deg_rewards.csv"
LOG_DEG      = "logs/policy_deg_rewards.csv"
LOG_LSTM     = "logs/policy_brady_deg_lstm_rewards.csv"
LOG_DEG_LSTM = "logs/policy_deg_lstm_rewards.csv"
LOG_EXTRAOBS = "logs/policy_brady_deg_lstm_extraobs_rewards.csv"

MLP_TIMESTEPS  = 1_000_000
LSTM_TIMESTEPS = 2_000_000
EVAL_TRIALS    = 32


# ── Helpers ──────────────────────────────────────────────────────────────────

def run(cmd: list, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"[pipeline] {label}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[pipeline] WARNING: '{label}' exited with code {result.returncode}")


def load_csv(path: str):
    """Load (timesteps, mean_reward) pairs from a reward CSV, or return empty arrays."""
    if not os.path.exists(path):
        return np.array([]), np.array([])
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if len(data) == 0:
        return np.array([]), np.array([])
    return data[:, 0], data[:, 1]


# ── Stage 1: Train all missing policies ──────────────────────────────────────

def _train_if_missing(policy_path: str, cmd_args: list, timesteps: int, label: str) -> None:
    if os.path.exists(policy_path):
        print(f"[pipeline] Skipping: {policy_path} already exists.")
        return
    run([PYTHON, "train_exo.py"] + cmd_args + ["--timesteps", str(timesteps)], label)


def stage_train():
    print("\n" + "=" * 60)
    print("[pipeline] STAGE 1 — Train missing policies")
    print(f"  MLP  budget: {MLP_TIMESTEPS:,} steps")
    print(f"  LSTM budget: {LSTM_TIMESTEPS:,} steps")
    print("=" * 60)

    # MLP policies (1M steps)
    _train_if_missing(POLICY_BRADY, [],
                      MLP_TIMESTEPS, "Train policy_brady_deg (MLP)")
    _train_if_missing(POLICY_DEG,   ["--no-bradykinesia"],
                      MLP_TIMESTEPS, "Train policy_deg (MLP)")

    # LSTM policies (2M steps)
    _train_if_missing(POLICY_LSTM,     ["--lstm"],
                      LSTM_TIMESTEPS, "Train policy_brady_deg_lstm")
    _train_if_missing(POLICY_DEG_LSTM, ["--no-bradykinesia", "--lstm"],
                      LSTM_TIMESTEPS, "Train policy_deg_lstm")
    _train_if_missing(POLICY_EXTRAOBS, ["--lstm", "--extraobs"],
                      LSTM_TIMESTEPS, "Train policy_brady_deg_lstm_extraobs")


# ── Stage 2: Combined reward chart ───────────────────────────────────────────

def stage_chart():
    print("\n" + "=" * 60)
    print("[pipeline] STAGE 2 — Combined reward curves")
    print("=" * 60)

    series = [
        (LOG_BRADY,    "policy_brady_deg (MLP)",        "C0", "-"),
        (LOG_DEG,      "policy_deg (MLP)",              "C1", "-"),
        (LOG_LSTM,     "policy_brady_deg_lstm",         "C2", "--"),
        (LOG_DEG_LSTM, "policy_deg_lstm",               "C3", "--"),
        (LOG_EXTRAOBS, "policy_brady_deg_lstm_extraobs","C4", ":"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    plotted = False
    for csv_path, label, color, linestyle in series:
        ts, rwd = load_csv(csv_path)
        if len(ts) == 0:
            print(f"  [chart] No data for {label} — skipping.")
            continue
        ax.plot(ts, rwd, label=label, color=color, linestyle=linestyle, linewidth=1.5)
        plotted = True

    if plotted:
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("Training Reward Curves — All Models")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        os.makedirs("results/pipeline", exist_ok=True)
        out = "results/pipeline/reward_curves_all_models.png"
        plt.savefig(out, dpi=150)
        print(f"  [chart] Saved → {out}")
    else:
        print("  [chart] No CSV logs found — skipping chart.")
    plt.close()


# ── Stage 3: Per-policy evaluations ──────────────────────────────────────────

def stage_evaluate():
    print("\n" + "=" * 60)
    print("[pipeline] STAGE 3 — Per-policy evaluations")
    print("=" * 60)

    evals = [
        # (policy_path, is_extraobs)
        (POLICY_BRADY,    False),
        (POLICY_DEG,      False),
        (POLICY_LSTM,     False),
        (POLICY_DEG_LSTM, False),
        (POLICY_EXTRAOBS, True),
    ]

    for policy_path, extraobs in evals:
        if not os.path.exists(policy_path):
            print(f"  [eval] Skipping missing policy: {policy_path}")
            continue
        basename = os.path.splitext(os.path.basename(policy_path))[0]
        out_dir  = f"results/eval_{basename}"
        cmd = [
            PYTHON, "evaluation.py",
            "--exo-path",   policy_path,
            "--healthy-path", HEALTHY_PATH,
            "--trials",     str(EVAL_TRIALS),
            "--out-dir",    out_dir,
        ]
        if extraobs:
            cmd.append("--extraobs")
        run(cmd, f"Evaluate {basename}")


# ── Stage 4: Ablation comparisons ────────────────────────────────────────────

def stage_compare():
    print("\n" + "=" * 60)
    print("[pipeline] STAGE 4 — Ablation comparisons")
    print("=" * 60)

    required = [HEALTHY_PATH, POLICY_BRADY, POLICY_LSTM, POLICY_EXTRAOBS]
    missing  = [p for p in required if not os.path.exists(p)]
    if missing:
        print("[pipeline] ERROR — cannot run comparisons, missing:")
        for m in missing:
            print(f"  {m}")
        return

    run([
        PYTHON, "compare.py",
        "--healthy",      HEALTHY_PATH,
        "--brady",        POLICY_BRADY,
        "--lstm",         POLICY_LSTM,
        "--deg-lstm",     POLICY_DEG_LSTM,
        "--extraobs-pol", POLICY_EXTRAOBS,
    ], "All ablation comparisons")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("reGainX — Full Pipeline")
    print(f"  Stage 1: Train any missing policies")
    print(f"           MLP  (policy_brady_deg, policy_deg)          → {MLP_TIMESTEPS:,} steps")
    print(f"           LSTM (policy_*_lstm, policy_*_lstm_extraobs) → {LSTM_TIMESTEPS:,} steps")
    print(f"           (already-present policies are skipped)")
    print("  Stage 2: Combined reward curves chart")
    print("  Stage 3: Per-policy evaluation (32 trials each)")
    print("  Stage 4: Ablation comparisons (LSTM / Brady / ExtraObs)")
    print("=" * 60)

    for path in [HEALTHY_PATH]:
        if not os.path.exists(path):
            print(f"\nERROR: Required file not found: {path}")
            print("  Run train_healthy.py first.")
            return

    stage_train()
    stage_chart()
    stage_evaluate()
    stage_compare()

    print("\n" + "=" * 60)
    print("[pipeline] Complete.")
    print("  results/pipeline/reward_curves_all_models.png")
    print("  results/eval_*/")
    print("  results/lstm_ablation/")
    print("  results/bradykinesia_ablation/")
    print("  results/extraobs_ablation/")
    print("=" * 60)


if __name__ == "__main__":
    main()
