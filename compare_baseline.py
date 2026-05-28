"""
Baseline vs RecPPO policy comparison on the brady+deg environment.

Evaluates two policies on identical 80 episodes (5 per cell of a 4×4
angle-bin × severity-quartile grid):
  - Baseline : policy_deg.zip (MLP, deg-only trained — brady is OOD for it)
  - RecPPO   : policy_brady_deg_recurrent.zip (trained with brady+deg)

Outputs to results/baseline_comparison/:
  confusion_matrix_baseline.png
  confusion_matrix_recppo.png
  confusion_matrix_no_exo.png
  comparison_metrics.png          (4-panel combined figure)
  pearsonr_by_severity.png        (standalone severity-sorted line plot)

Usage:
    python compare_baseline.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import myosuite  # noqa: F401 — registers envs
from myosuite.utils import gym
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT_PPO = True
except ImportError:
    _HAS_RECURRENT_PPO = False

from envs.elbow_env import CombinedExoOnlyWrapper
from evaluation import plan_trials, severity_quartile_to_range, angle_bin_to_target
from utils import (
    plot_confusion_matrix,
    compute_severity,
    get_angle_bin,
    get_severity_quartile,
)

# ---------------------------------------------------------------------------
# Hardcoded policy paths
# ---------------------------------------------------------------------------
HEALTHY_PATH   = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\healthy_policy.zip"
BASELINE_PATH  = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\policy_deg.zip"
RECURRENT_PATH = r"C:\Users\rohan\Downloads\ML\ReGainX\policies\policy_brady_deg_recurrent.zip"
OUT_DIR        = "results/baseline_comparison"

# ---------------------------------------------------------------------------
# Evaluation constants
# ---------------------------------------------------------------------------
N_EPISODES     = 80          # 5 per cell × 16 cells
MAX_STEPS      = 500
ANGLE_BINS     = 4
SEVERITY_BINS  = 4
ANGLE_LABELS   = ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5"]
SEVERITY_LABELS = ["Q1 mild", "Q2", "Q3", "Q4 severe"]
