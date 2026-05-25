# EMG Noise Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify `NoisyExoWrapper` from a bias+jitter noise model to the standard RL pattern — per-episode sigma, fresh zero-mean white noise per step.

**Architecture:** Remove the `_noise_bias` field, `_reset_noise_state()` method, and `jitter_scale` parameter from `NoisyExoWrapper`. Replace `_inject_noise()` with a single `rng.normal(0, sigma)` draw per step. Public API and all callers remain unchanged.

**Tech Stack:** Python, NumPy, Gymnasium, pytest

---

## File Map

- Modify: `envs/elbow_env_noisy.py` — remove bias/jitter, simplify to white noise
- Create: `tests/test_noisy_env.py` — unit tests for the new noise model (no myosuite required)

---

### Task 1: Write Failing Tests

**Files:**
- Create: `tests/test_noisy_env.py`

- [ ] **Step 1: Create the test file**

```python
"""
Unit tests for NoisyExoWrapper — no myosuite required.
Uses MockInnerEnv to satisfy the wrapper's constructor.
"""

import inspect
import numpy as np
import gymnasium as gym
import pytest

from envs.elbow_env_noisy import NoisyExoWrapper

N_ACT     = 6
OBS_DIM   = 8
ACT_SLICE = slice(2, 8)


class MockInnerEnv(gym.Env):
    """Minimal flat-obs env: obs = np.full(8, 0.5). No myosuite needed."""

    def __init__(self, obs_dim: int = OBS_DIM):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self._obs = np.full(obs_dim, 0.5, dtype=np.float32)

    def reset(self, **kwargs):
        return self._obs.copy(), {}

    def step(self, action):
        return self._obs.copy(), 0.0, False, False, {}


# ── Constructor tests ────────────────────────────────────────────────────────

def test_no_jitter_scale_parameter():
    """jitter_scale must not exist in the constructor signature."""
    sig = inspect.signature(NoisyExoWrapper.__init__)
    assert "jitter_scale" not in sig.parameters, (
        "jitter_scale was not removed from NoisyExoWrapper.__init__"
    )


def test_no_noise_bias_attribute():
    """_noise_bias must not exist on the wrapper instance."""
    env = NoisyExoWrapper(MockInnerEnv(), noise_sigma=0.05, rng_seed=0)
    assert not hasattr(env, "_noise_bias"), (
        "_noise_bias attribute was not removed from NoisyExoWrapper"
    )


# ── Noise model tests ────────────────────────────────────────────────────────

def test_noise_zero_mean_within_episode():
    """
    Over 500 steps in one episode the mean noise per channel must be near zero.
    White noise: E[noise_t] = 0. With 500 draws, |mean| < 0.05 * sigma (3-sigma bound).
    """
    sigma = 0.10
    env   = NoisyExoWrapper(MockInnerEnv(), noise_sigma=sigma, rng_seed=42)
    env.reset()
    clean = np.full(OBS_DIM, 0.5, dtype=np.float32)

    noisy_readings = []
    for _ in range(500):
        obs, *_ = env.step(np.array([0.0], dtype=np.float32))
        noisy_readings.append(obs[ACT_SLICE] - clean[ACT_SLICE])

    per_channel_mean = np.abs(np.mean(noisy_readings, axis=0))
    threshold = 0.05 * sigma  # generous: allows up to 5% of sigma as residual mean
    assert np.all(per_channel_mean < threshold), (
        f"Noise is not zero-mean. Per-channel |mean|={per_channel_mean}, threshold={threshold}"
    )


def test_sigma_constant_within_episode():
    """_current_sigma must not change between steps within an episode."""
    env = NoisyExoWrapper(
        MockInnerEnv(), noise_sigma=0.05, randomize_sigma=True,
        sigma_low=0.01, sigma_high=0.10, rng_seed=7,
    )
    env.reset()
    sigma_at_reset = env._current_sigma

    for _ in range(100):
        env.step(np.array([0.0], dtype=np.float32))
        assert env._current_sigma == sigma_at_reset, (
            f"_current_sigma changed mid-episode: expected {sigma_at_reset}, "
            f"got {env._current_sigma}"
        )


def test_noise_only_on_activation_channels():
    """Channels 0 and 1 (qpos, qvel) must be untouched by noise injection."""
    env = NoisyExoWrapper(MockInnerEnv(), noise_sigma=0.10, rng_seed=1)
    env.reset()

    for _ in range(50):
        obs, *_ = env.step(np.array([0.0], dtype=np.float32))
        assert obs[0] == pytest.approx(0.5, abs=1e-6), "qpos (index 0) was noised"
        assert obs[1] == pytest.approx(0.5, abs=1e-6), "qvel (index 1) was noised"


def test_sigma_randomized_across_episodes():
    """With randomize_sigma=True, sigma must differ across at least some episodes."""
    env = NoisyExoWrapper(
        MockInnerEnv(), randomize_sigma=True,
        sigma_low=0.01, sigma_high=0.10, rng_seed=99,
    )
    sigmas = set()
    for _ in range(20):
        env.reset()
        sigmas.add(round(env._current_sigma, 6))

    assert len(sigmas) > 1, (
        "randomize_sigma=True produced the same sigma across 20 episodes"
    )


def test_obs_clipped_to_bounds():
    """Even with large sigma the noised obs must stay in [0, 1]."""
    env = NoisyExoWrapper(MockInnerEnv(), noise_sigma=1.0, rng_seed=3)
    env.reset()

    for _ in range(200):
        obs, *_ = env.step(np.array([0.0], dtype=np.float32))
        assert np.all(obs[ACT_SLICE] >= 0.0), "obs below 0 after clipping"
        assert np.all(obs[ACT_SLICE] <= 1.0), "obs above 1 after clipping"


def test_zero_sigma_passthrough():
    """noise_sigma=0 must return the observation completely unchanged."""
    env   = NoisyExoWrapper(MockInnerEnv(), noise_sigma=0.0, rng_seed=0)
    obs0, _ = env.reset()

    assert np.allclose(obs0, 0.5), "reset obs changed with sigma=0"

    for _ in range(20):
        obs, *_ = env.step(np.array([0.0], dtype=np.float32))
        assert np.allclose(obs, 0.5), "step obs changed with sigma=0"
```

- [ ] **Step 2: Run tests to confirm they fail with the current implementation**

```
pytest tests/test_noisy_env.py -v
```

Expected output (all fail or errors):
- `test_no_jitter_scale_parameter` — FAIL (`jitter_scale` still present)
- `test_no_noise_bias_attribute` — FAIL (`_noise_bias` still exists)
- `test_noise_zero_mean_within_episode` — may PASS (bias shifts mean) or FAIL
- `test_sigma_constant_within_episode` — PASS (unchanged by this task)
- All others — mixed; the key gate is the first two fail confirming old code is present.

- [ ] **Step 3: Commit the test file**

```bash
git add tests/test_noisy_env.py
git commit -m "test: add unit tests for simplified NoisyExoWrapper (white noise model)"
```

---

### Task 2: Simplify `envs/elbow_env_noisy.py`

**Files:**
- Modify: `envs/elbow_env_noisy.py`

- [ ] **Step 1: Replace the entire file with the simplified implementation**

Write `envs/elbow_env_noisy.py` with the content below. Every section is shown in full — no ellipsis.

```python
"""
NoisyExoWrapper — injects additive white Gaussian noise into the muscle-activation
channels of the exo observation to simulate surface EMG measurement noise.

EMG background
--------------
Surface EMG signals are normalised to [0, 1] after rectification and smoothing.
The dominant noise source is additive white Gaussian noise (AWGN). Typical SNR:
  - High quality recording : ~34 dB  →  sigma ≈ 0.010
  - Mid-range (realistic)  : ~20 dB  →  sigma ≈ 0.050
  - Noisy recording        : ~14 dB  →  sigma ≈ 0.100

(SNR in dB = 20·log10(0.5 / sigma), assuming mean activation ≈ 0.5)

Noise model (standard RL / domain randomisation pattern)
---------------------------------------------------------
At reset:   sigma ~ Uniform(sigma_low, sigma_high)   if randomize_sigma=True
            sigma = noise_sigma                        otherwise

Each step:  noise_t ~ N(0, sigma),  shape (6,)
            obs[2:8] += noise_t
            obs[2:8]  = clip(obs[2:8], act_low, act_high)

Sigma is constant within an episode. Noise values are fresh, independent, and
zero-mean every step — the LSTM cannot cancel zero-mean noise, so it must
develop genuine robustness rather than learning to track a fixed bias.

Observation layout (hide_pose_err=True, extra_obs=False → 8-D flat obs):
  Index 0    : qpos  — elbow joint angle    (physical sensor, no noise)
  Index 1    : qvel  — elbow joint velocity  (physical sensor, no noise)
  Indices 2–7: act   — 6 muscle activations  (EMG-derived, noise injected here)
"""

import numpy as np
import gymnasium as gym

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.temporal_buffer import TemporalStackWrapper

ACT_SLICE = slice(2, 8)   # muscle activation indices in the flat 8-D obs
N_ACT     = 6             # number of muscle channels


class NoisyExoWrapper(gym.Wrapper):
    """
    Wraps a CombinedExoOnlyWrapper (or a TemporalStackWrapper around one)
    and injects additive white Gaussian noise into the 6 muscle activation
    channels to simulate surface EMG measurement noise.

    Noise model: fresh N(0, sigma) drawn independently every step.
    Sigma is fixed for the duration of the episode.

    Parameters
    ----------
    env:
        CombinedExoOnlyWrapper, or TemporalStackWrapper(CombinedExoOnlyWrapper).
    noise_sigma:
        Episode noise standard deviation. 0.0 = clean. Ignored when
        randomize_sigma=True.
    randomize_sigma:
        If True, sigma is re-sampled from Uniform(sigma_low, sigma_high) at
        the start of every episode. Use for domain-randomisation training.
    sigma_low, sigma_high:
        Bounds for per-episode sigma randomisation.
        Default (0.01, 0.10) → SNR ≈ 34–14 dB (realistic surface EMG range).
    rng_seed:
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        env,
        noise_sigma: float    = 0.05,
        randomize_sigma: bool = False,
        sigma_low: float      = 0.01,
        sigma_high: float     = 0.10,
        rng_seed: int         = None,
    ):
        super().__init__(env)
        self.noise_sigma     = noise_sigma
        self.randomize_sigma = randomize_sigma
        self.sigma_low       = sigma_low
        self.sigma_high      = sigma_high
        self._rng            = np.random.default_rng(rng_seed)

        self._current_sigma  = noise_sigma   # updated at each reset

        # Unpack wrapper chain to reach the CombinedExoOnlyWrapper
        if isinstance(env, TemporalStackWrapper):
            self._is_lstm   = True
            self._inner_exo = env.env
        else:
            self._is_lstm   = False
            self._inner_exo = env

        # Per-channel clip bounds from the underlying flat obs space
        flat_space     = self._inner_exo.observation_space
        self._act_low  = flat_space.low[ACT_SLICE].copy()    # shape (6,)
        self._act_high = flat_space.high[ACT_SLICE].copy()   # shape (6,)

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _inject_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add N(0, sigma) noise to muscle-activation channels and clip."""
        if self._current_sigma == 0.0:
            return obs

        noised = obs.copy()

        if self._is_lstm:
            # obs shape: (1, window, obs_dim) — each frame gets independent noise
            noise = self._rng.normal(
                0.0, self._current_sigma,
                size=noised[..., ACT_SLICE].shape,
            ).astype(np.float32)
            noised[..., ACT_SLICE] += noise
            noised[..., ACT_SLICE]  = np.clip(
                noised[..., ACT_SLICE], self._act_low, self._act_high,
            )
        else:
            noise = self._rng.normal(
                0.0, self._current_sigma, size=N_ACT,
            ).astype(np.float32)
            noised[ACT_SLICE] += noise
            noised[ACT_SLICE]  = np.clip(
                noised[ACT_SLICE], self._act_low, self._act_high,
            )

        return noised

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        if self.randomize_sigma:
            self._current_sigma = float(
                self._rng.uniform(self.sigma_low, self.sigma_high)
            )
        else:
            self._current_sigma = self.noise_sigma

        obs, info = self.env.reset(**kwargs)
        return self._inject_noise(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._inject_noise(obs), reward, done, truncated, info
```

- [ ] **Step 2: Run tests to confirm they all pass**

```
pytest tests/test_noisy_env.py -v
```

Expected output — all 8 tests PASS:
```
tests/test_noisy_env.py::test_no_jitter_scale_parameter PASSED
tests/test_noisy_env.py::test_no_noise_bias_attribute PASSED
tests/test_noisy_env.py::test_noise_zero_mean_within_episode PASSED
tests/test_noisy_env.py::test_sigma_constant_within_episode PASSED
tests/test_noisy_env.py::test_noise_only_on_activation_channels PASSED
tests/test_noisy_env.py::test_sigma_randomized_across_episodes PASSED
tests/test_noisy_env.py::test_obs_clipped_to_bounds PASSED
tests/test_noisy_env.py::test_zero_sigma_passthrough PASSED
```

- [ ] **Step 3: Commit the implementation**

```bash
git add envs/elbow_env_noisy.py
git commit -m "refactor: simplify NoisyExoWrapper to standard RL white noise pattern

Remove bias+jitter model. Each step draws fresh N(0,sigma); sigma is fixed
per episode. Removes _noise_bias, _reset_noise_state(), and jitter_scale.
LSTM cannot cancel zero-mean noise, forcing genuine robustness training."
```

---

## Spec Coverage Check

- [x] Remove `_noise_bias` field and initialisation — Task 2
- [x] Remove `jitter_scale` constructor parameter and field — Task 2
- [x] Remove `_reset_noise_state()` method — Task 2
- [x] Remove all bias/jitter logic in `_inject_noise()` — Task 2
- [x] Keep `noise_sigma, randomize_sigma, sigma_low, sigma_high, rng_seed` — Task 2
- [x] Per-episode sigma sampling in `reset()` — Task 2
- [x] Flat obs support (8,) and LSTM obs support (1, window, 8) — Task 2
- [x] Clipping to `[act_low, act_high]` — Task 2
- [x] `ACT_SLICE = slice(2, 8)` unchanged — Task 2
- [x] `test_no_jitter_scale_parameter` — Task 1
- [x] `test_no_noise_bias_attribute` — Task 1
- [x] `test_noise_zero_mean_within_episode` — Task 1
- [x] `test_sigma_constant_within_episode` — Task 1
- [x] `test_noise_only_on_activation_channels` — Task 1
- [x] `test_sigma_randomized_across_episodes` — Task 1
- [x] `test_obs_clipped_to_bounds` — Task 1
- [x] `test_zero_sigma_passthrough` — Task 1
- [x] No other scripts need updating (`noise_eval.py`, `noise_impact.py`, `train_recurrent_noisy.py`, `pipeline.py`) — confirmed in spec; no tasks added
