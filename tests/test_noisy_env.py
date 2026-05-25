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
