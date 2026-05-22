"""
Full reGainX pipeline: train → chart → evaluate → compare.

Trains the three CNN-LSTM policies (2M steps each), skipping already-trained
MLP models. Generates a combined reward-curve chart for all models, runs
evaluation.py for each policy, then runs all three ablation comparisons
via compare.py.

Usage:
    python pipeline.py

Requires:
    policies/healthy_policy.zip       (train_healthy.py)
    policies/policy_brady_deg.zip     (train_exo.py)
    policies/policy_deg.zip           (train_exo.py --no-bradykinesia)

Trains (if not present):
    policies/policy_brady_deg_lstm.zip
    policies/policy_deg_lstm.zip
    policies/policy_brady_deg_lstm_extraobs.zip

Outputs:
    results/pipeline/reward_curves_all_models.png
    results/eval_policy_*/                 (per-policy evaluation)
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


# ── Stage 1: Train LSTM policies ─────────────────────────────────────────────

def stage_train():
    print("\n" + "=" * 60)
    print("[pipeline] STAGE 1 — Train LSTM policies")
    print("=" * 60)

    if not os.path.exists(POLICY_LSTM):
        run([PYTHON, "train_exo.py", "--lstm", "--timesteps", str(LSTM_TIMESTEPS)],
            "Train policy_brady_deg_lstm")
    else:
        print(f"[pipeline] Skipping: {POLICY_LSTM} already exists.")

    if not os.path.exists(POLICY_DEG_LSTM):
        run([PYTHON, "train_exo.py", "--no-bradykinesia", "--lstm",
             "--timesteps", str(LSTM_TIMESTEPS)],
            "Train policy_deg_lstm")
    else:
        print(f"[pipeline] Skipping: {POLICY_DEG_LSTM} already exists.")

    if not os.path.exists(POLICY_EXTRAOBS):
        run([PYTHON, "train_exo.py", "--lstm", "--extraobs",
             "--timesteps", str(LSTM_TIMESTEPS)],
            "Train policy_brady_deg_lstm_extraobs")
    else:
        print(f"[pipeline] Skipping: {POLICY_EXTRAOBS} already exists.")


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
    print("  Stage 1: Train LSTM policies (2M steps each, skip if present)")
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
