import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MixPool2d(nn.Module):
    """Equal-weight max + average adaptive pooling (SWCTNet 2025)."""

    def __init__(self, output_size):
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return 0.5 * self.max_pool(x) + 0.5 * self.avg_pool(x)


class TemporalCNNExtractor(BaseFeaturesExtractor):
    """
    2D temporal CNN feature extractor for SB3 PPO.

    Observation space: Box(shape=(1, 20, 13)) — channels first.
    Treats the 20-step buffer as a single-channel image of shape (20, 13).

    Architecture (SWCTNet-inspired, PMC 2025):
      - ELU activations for smooth gradients on biomechanical data
      - BatchNorm2d for training stability
      - MixPool2d (max+avg) to retain temporal peaks
      - Dropout(0.3) before FC

    SB3 usage in train_exo.py:
      policy_kwargs = dict(
          features_extractor_class=TemporalCNNExtractor,
          features_extractor_kwargs=dict(features_dim=256),
          net_arch=dict(pi=[256, 128], vf=[256, 128]),
      )
      PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, ...)
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]  # = 1

        self.cnn = nn.Sequential(
            # Block 1: preserve spatial dims
            nn.Conv2d(n_channels, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # Block 2: deeper features
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # Block 3: squeeze feature (width) dimension
            nn.Conv2d(64, 128, kernel_size=(3, 1)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # Pool to (batch, 128, 4, 1)
            MixPool2d(output_size=(4, 1)),
            nn.Dropout(0.3),
            nn.Flatten(),
        )

        # Dynamically compute flat dim to avoid hardcoding
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            flat_dim = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_dim, features_dim),
            nn.ELU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
