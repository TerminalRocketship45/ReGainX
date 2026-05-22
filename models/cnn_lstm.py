"""
CNN-LSTM feature extractor for SB3 PPO.

Architecture (Han et al. 2024, EMG+IMU elbow prediction):
  Conv1d extracts local temporal features from the observation window;
  LSTM carries long-range sequential memory across the window.

Input from TemporalStackWrapper: Box(shape=(1, window, obs_dim)).

Literature hyperparameters (SB3 rl-baselines3-zoo + exoskeleton papers):
  learning_rate=3e-4, n_steps=2048, batch_size=64,
  gamma=0.99, ent_coef=0.0, clip_range=0.2
"""

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CNNLSTMExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # observation_space.shape = (1, window, obs_dim)
        _, window, obs_dim = observation_space.shape

        # Conv1d: treat obs_dim as channels, window as sequence length
        self.cnn = nn.Sequential(
            nn.Conv1d(obs_dim, 64,  kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64,  128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )

        # LSTM: processes CNN output sequence, returns final hidden state
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ELU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # (batch, 1, window, obs_dim) → (batch, window, obs_dim)
        x = observations.squeeze(1)
        # (batch, obs_dim, window) — Conv1d channels-first
        x = x.permute(0, 2, 1)
        x = self.cnn(x)                    # (batch, 128, window)
        x = x.permute(0, 2, 1)            # (batch, window, 128)
        _, (h_n, _) = self.lstm(x)        # h_n: (num_layers, batch, 256)
        x = h_n[-1]                        # (batch, 256) last-layer hidden state
        return self.head(x)                # (batch, features_dim)
