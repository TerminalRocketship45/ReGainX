"""
NoisyExoWrapper — injects white Gaussian noise into the muscle-activation
channels of the exo observation to simulate surface EMG measurement noise.

EMG background
--------------
Surface EMG signals are normalised to [0, 1] after rectification and
smoothing.  The dominant noise source is additive white Gaussian noise
(AWGN).  Typical SNR values reported in the literature:
  - High quality recording : ~30 dB  →  sigma ≈ 0.016
  - Mid-range (realistic)  : ~20 dB  →  sigma ≈ 0.050
  - Noisy recording        : ~15 dB  →  sigma ≈ 0.089
  - Poor quality / clinical: ~10 dB  →  sigma ≈ 0.158

(SNR in dB = 20·log10(0.5 / sigma), assuming mean activation ≈ 0.5)

Observation layout (hide_pose_err=True, extra_obs=False → 8-D flat obs):
  Index 0    : qpos  — elbow joint angle   (physical sensor, no EMG noise)
  Index 1    : qvel  — elbow joint velocity (physical sensor, no EMG noise)
  Indices 2–7: act   — 6 muscle activations (EMG-derived, noise injected here)

Noise is drawn freshly each timestep from N(0, sigma) and clipped to the
underlying observation-space bounds after injection.

For the CNN-LSTM case the wrapped env is a TemporalStackWrapper and the
obs shape is (1, window, obs_dim).  Noise is still applied to the last
dimension's muscle-activation slice, independently at every step.
"""

import numpy as np
import gymnasium as gym

from envs.elbow_env import CombinedExoOnlyWrapper
from envs.temporal_buffer import TemporalStackWrapper

ACT_SLICE = slice(2, 8)  # indices of muscle activations in the flat 8-D obs


class NoisyExoWrapper(gym.Wrapper):
    """
    Wraps a CombinedExoOnlyWrapper (or a TemporalStackWrapper around one)
    and adds additive white Gaussian noise to the 6 muscle activation channels.

    Parameters
    ----------
    env:
        CombinedExoOnlyWrapper, or TemporalStackWrapper(CombinedExoOnlyWrapper).
    noise_sigma:
        Noise standard deviation (sigma) applied each step.  0.0 = clean.
        Ignored when randomize_sigma=True.
    randomize_sigma:
        If True, sigma is re-sampled from Uniform(sigma_low, sigma_high) at
        the start of every episode.  Use this for domain-randomisation training.
    sigma_low, sigma_high:
        Bounds for per-episode sigma randomisation.
        Default: (0.01, 0.10) → SNR ≈ 34–14 dB (realistic surface EMG).
    rng_seed:
        Optional seed for the noise RNG.
    """

    def __init__(
        self,
        env,
        noise_sigma: float = 0.05,
        randomize_sigma: bool = False,
        sigma_low: float = 0.01,
        sigma_high: float = 0.10,
        rng_seed: int = None,
    ):
        super().__init__(env)
        self.noise_sigma      = noise_sigma
        self.randomize_sigma  = randomize_sigma
        self.sigma_low        = sigma_low
        self.sigma_high       = sigma_high
        self._rng             = np.random.default_rng(rng_seed)
        self._current_sigma   = noise_sigma

        # Unpack to the CombinedExoOnlyWrapper for flat obs bounds
        if isinstance(env, TemporalStackWrapper):
            self._is_lstm = True
            self._inner_exo = env.env  # CombinedExoOnlyWrapper
        else:
            self._is_lstm = False
            self._inner_exo = env     # CombinedExoOnlyWrapper

        # Per-channel bounds for clipping (from the flat 8-D obs space)
        flat_space = self._inner_exo.observation_space
        self._act_low  = flat_space.low[ACT_SLICE].copy()   # shape (6,)
        self._act_high = flat_space.high[ACT_SLICE].copy()  # shape (6,)

    # ------------------------------------------------------------------

    def _sample_sigma(self) -> float:
        if self.randomize_sigma:
            return float(self._rng.uniform(self.sigma_low, self.sigma_high))
        return self.noise_sigma

    def _inject_noise(self, obs: np.ndarray) -> np.ndarray:
        """Add AWGN to muscle-activation channels and clip to valid range."""
        if self._current_sigma == 0.0:
            return obs
        noised = obs.copy()
        if self._is_lstm:
            # shape: (1, window, obs_dim) — noise each frame independently
            noise = self._rng.normal(
                0.0, self._current_sigma,
                size=noised[..., ACT_SLICE].shape,
            ).astype(np.float32)
            noised[..., ACT_SLICE] += noise
            noised[..., ACT_SLICE] = np.clip(
                noised[..., ACT_SLICE], self._act_low, self._act_high
            )
        else:
            noise = self._rng.normal(
                0.0, self._current_sigma, size=6,
            ).astype(np.float32)
            noised[ACT_SLICE] += noise
            noised[ACT_SLICE] = np.clip(
                noised[ACT_SLICE], self._act_low, self._act_high
            )
        return noised

    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        self._current_sigma = self._sample_sigma()
        obs, info = self.env.reset(**kwargs)
        return self._inject_noise(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._inject_noise(obs), reward, done, truncated, info
