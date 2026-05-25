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
