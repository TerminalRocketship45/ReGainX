import numpy as np
import gymnasium as gym
from collections import deque


class TemporalStackWrapper(gym.Wrapper):
    """
    Maintains a rolling deque of the last `window` observations.
    Returns stacked obs as (1, window, obs_dim) — channels-first for TemporalCNNExtractor.
    Only instantiated when --cnn flag is active in train_exo.py / compare.py.
    """

    def __init__(self, env, window: int = 20):
        super().__init__(env)
        assert len(env.observation_space.shape) == 1, (
            f"TemporalStackWrapper expects a 1-D observation space, got {env.observation_space.shape}"
        )
        self.window = window
        obs_dim = env.observation_space.shape[0]
        self._buffer: deque = deque(maxlen=window)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, window, obs_dim),
            dtype=np.float32,
        )

    def _stack(self) -> np.ndarray:
        assert len(self._buffer) == self.window, (
            f"Buffer has {len(self._buffer)} entries; call reset() before step()."
        )
        return np.array(self._buffer, dtype=np.float32)[np.newaxis]

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._buffer.clear()
        for _ in range(self.window):
            self._buffer.append(obs.astype(np.float32, copy=False))
        return self._stack(), info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        self._buffer.append(obs.astype(np.float32, copy=False))
        return self._stack(), reward, done, truncated, info
