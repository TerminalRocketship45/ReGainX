"""
Standalone inference latency benchmark for all saved policies.

Measures only model.predict() wall-clock time (no environment stepping).
Each model is warmed up with 50 calls then timed over 500 calls.

Usage:
    python latency_bench.py
"""

import time
import numpy as np
from stable_baselines3 import PPO

try:
    from sb3_contrib import RecurrentPPO
    _HAS_RECURRENT = True
except ImportError:
    _HAS_RECURRENT = False

from models.cnn_lstm import CNNLSTMExtractor  # noqa: F401  needed for PPO.load

WARMUP = 50
REPS   = 500

MODELS = [
    # (label, path, arch)
    # arch: "mlp" | "cnn" | "recurrent"
    ("Healthy (MLP)",              "policies/healthy_policy.zip",                  "mlp"),
    ("Brady+Deg MLP",              "policies/policy_brady_deg.zip",                "mlp"),
    ("Deg-only MLP",               "policies/policy_deg.zip",                      "mlp"),
    ("Brady+Deg CNN-LSTM",         "policies/policy_brady_deg_lstm.zip",           "cnn"),
    ("Deg-only CNN-LSTM",          "policies/policy_deg_lstm.zip",                 "cnn"),
    ("Brady+Deg CNN-LSTM +ExtraObs","policies/policy_brady_deg_lstm_extraobs.zip", "cnn"),
    ("Brady+Deg RecurrentPPO",     "policies/policy_brady_deg_recurrent.zip",      "recurrent"),
    ("Brady+Deg Recurrent (Noisy)","policies/policy_brady_deg_recurrent_noisy.zip","recurrent"),
    ("Deg-only RecurrentPPO",      "policies/policy_deg_recurrent.zip",            "recurrent"),
]


def bench_mlp(model: PPO, obs: np.ndarray, warmup: int, reps: int) -> np.ndarray:
    for _ in range(warmup):
        model.predict(obs, deterministic=True)
    times = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        model.predict(obs, deterministic=True)
        times[i] = (time.perf_counter() - t0) * 1000
    return times


def bench_recurrent(model, obs: np.ndarray, warmup: int, reps: int) -> np.ndarray:
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    for _ in range(warmup):
        _, lstm_states = model.predict(obs, state=lstm_states,
                                       episode_start=ep_start, deterministic=True)
        ep_start = np.zeros((1,), dtype=bool)
    lstm_states = None
    ep_start = np.ones((1,), dtype=bool)
    times = np.empty(reps)
    for i in range(reps):
        t0 = time.perf_counter()
        _, lstm_states = model.predict(obs, state=lstm_states,
                                       episode_start=ep_start, deterministic=True)
        times[i] = (time.perf_counter() - t0) * 1000
        ep_start = np.zeros((1,), dtype=bool)
    return times


def stats(times: np.ndarray) -> dict:
    return {
        "mean":   float(np.mean(times)),
        "std":    float(np.std(times)),
        "min":    float(np.min(times)),
        "p5":     float(np.percentile(times, 5)),
        "median": float(np.median(times)),
        "p95":    float(np.percentile(times, 95)),
        "p99":    float(np.percentile(times, 99)),
        "max":    float(np.max(times)),
    }


def main():
    results = []
    col_w = 28

    print(f"\n{'Model':<{col_w}}  {'mean':>7}  {'std':>6}  {'min':>6}  {'p5':>6}  "
          f"{'median':>7}  {'p95':>7}  {'p99':>7}  {'max':>7}   (ms)")
    print("-" * 100)

    for label, path, arch in MODELS:
        if arch == "recurrent" and not _HAS_RECURRENT:
            print(f"{label:<{col_w}}  [SKIP — sb3_contrib not installed]")
            continue

        try:
            if arch == "recurrent":
                model = RecurrentPPO.load(path)
            else:
                model = PPO.load(path)
        except Exception as e:
            print(f"{label:<{col_w}}  [LOAD ERROR: {e}]")
            continue

        obs_shape = model.observation_space.shape
        obs = model.observation_space.sample().astype(np.float32)
        obs = obs.reshape(1, *obs_shape) if arch != "recurrent" else obs

        try:
            if arch == "recurrent":
                times = bench_recurrent(model, obs, WARMUP, REPS)
            else:
                times = bench_mlp(model, obs, WARMUP, REPS)
        except Exception as e:
            print(f"{label:<{col_w}}  [BENCH ERROR: {e}]")
            continue

        s = stats(times)
        results.append({"label": label, "arch": arch, "obs_shape": obs_shape, **s})

        print(f"{label:<{col_w}}  {s['mean']:>7.3f}  {s['std']:>6.3f}  {s['min']:>6.3f}  "
              f"{s['p5']:>6.3f}  {s['median']:>7.3f}  {s['p95']:>7.3f}  {s['p99']:>7.3f}  "
              f"{s['max']:>7.3f}")

    print("-" * 100)

    # Group summary
    print("\n=== Architecture Summary ===")
    for arch_name in ["mlp", "cnn", "recurrent"]:
        group = [r for r in results if r["arch"] == arch_name]
        if not group:
            continue
        all_means = [r["mean"] for r in group]
        all_p99s  = [r["p99"]  for r in group]
        print(f"  {arch_name.upper():<12}  mean-of-means={np.mean(all_means):.3f} ms  "
              f"max-p99={max(all_p99s):.3f} ms  (n={len(group)} models)")

    print(f"\nWarmup={WARMUP} calls, Measured={REPS} calls per model.")


if __name__ == "__main__":
    main()
