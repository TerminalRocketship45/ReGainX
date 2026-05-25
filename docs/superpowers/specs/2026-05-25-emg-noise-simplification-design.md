# EMG Noise Simplification — Design Spec
**Date:** 2026-05-25
**Goal:** Simplify `NoisyExoWrapper` to the standard RL observation noise pattern for sim-to-real robustness training.

---

## Context

`NoisyExoWrapper` currently injects two-component noise into the 6 muscle activation channels of the exo observation:

```
noise_t = bias + jitter_t
bias    ~ N(0, sigma)           # sampled once per episode, per channel
jitter_t ~ N(0, sigma * 0.15)  # sampled every step
```

A literature review confirmed that:
- Per-episode sigma sampling is correct and research-backed (OpenAI DR, Isaac Lab, Daniel Takeshi's domain randomization guide)
- The **bias + jitter structure is non-standard** in RL robustness literature and has no research validation for sim-to-real
- The `jitter_scale=0.15` parameter has no empirical basis
- For sim-to-real robustness (the project goal), the standard pattern — **fixed sigma per episode, fresh zero-mean white noise per step** — is more appropriate

The bias component is actively counterproductive for RecurrentPPO: the LSTM can learn to estimate and cancel a constant per-episode offset, which means it solves a tracking problem rather than genuinely becoming noise-robust. Zero-mean white noise cannot be cancelled, so the LSTM must develop true robustness.

---

## Noise Model (After)

```
At reset:    sigma ~ Uniform(sigma_low, sigma_high)   # if randomize_sigma=True
             sigma = noise_sigma                        # if randomize_sigma=False

Each step:   noise_t ~ N(0, sigma),  shape (6,)
             obs[2:8] += noise_t
             obs[2:8]  = clip(obs[2:8], act_low, act_high)
```

Sigma is constant within an episode. Noise values are fresh, independent, and zero-mean every step.

---

## Changes

### `envs/elbow_env_noisy.py`

**Remove:**
- `_noise_bias` field and its initialisation
- `jitter_scale` constructor parameter and field
- `_reset_noise_state()` method
- All bias/jitter logic in `_inject_noise()`

**Keep:**
- `noise_sigma`, `randomize_sigma`, `sigma_low`, `sigma_high`, `rng_seed` — identical interface
- Per-episode sigma sampling in `reset()`
- Flat obs support (shape `(8,)`) and LSTM obs support (shape `(1, window, 8)`)
- Clipping to `[act_low, act_high]` after noise injection
- `ACT_SLICE = slice(2, 8)` — noise applied only to muscle activation channels

**Result:** `_inject_noise()` becomes a single `rng.normal(0, sigma, size=6)` draw per step.

### Constructor signature (after)

```python
NoisyExoWrapper(
    env,
    noise_sigma: float   = 0.05,
    randomize_sigma: bool = False,
    sigma_low: float     = 0.01,   # ~34 dB SNR
    sigma_high: float    = 0.10,   # ~14 dB SNR
    rng_seed: int        = None,
)
```

`jitter_scale` is removed. No other scripts need updating — it was never passed externally.

---

## Parameters

| Parameter | Value | SNR equivalent |
|---|---|---|
| `sigma_low` | 0.01 | ~34 dB (clean lab) |
| `sigma_high` | 0.10 | ~14 dB (noisy clinical) |
| Training default (randomized) | Uniform(0.01, 0.10) | 14–34 dB per episode |
| Mid-range (noise_eval study B) | 0.05 | ~20 dB |

SNR computed as `20 * log10(0.5 / sigma)` for a normalised EMG signal with mean amplitude ≈ 0.5.

---

## What Does Not Change

- `noise_eval.py` — no changes needed; uses `NoisyExoWrapper` via the same public API
- `noise_impact.py` — no changes needed
- `train_recurrent_noisy.py` — no changes needed; `randomize_sigma=True, sigma_low=0.01, sigma_high=0.10` remain valid
- `pipeline.py` — no changes needed
- `envs/__init__.py` — no changes needed

---

## Research Basis

- Daniel Takeshi, *Domain Randomization Tips* (2019): sample noise parameters at episode start, keep fixed for episode
- OpenAI dexterous manipulation: per-episode sigma, fresh N(0,σ) per step
- Isaac Lab: additive Gaussian observation noise with constant std during episode
- PMC (signal-dependent EMG noise): noise proportional to activation — noted but deferred; additive white noise is sufficient for robustness goal
- Sigma range: EMG literature reports electrode noise at 2–40% of max burst amplitude; [0.01, 0.10] covers the realistic range for normalised signals
