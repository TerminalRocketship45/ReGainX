import torch
import numpy as np
import gymnasium as gym
import pytest


def test_mixpool_output_shape():
    from models.temporal_cnn import MixPool2d
    pool = MixPool2d(output_size=(4, 1))
    x = torch.randn(2, 128, 18, 13)
    out = pool(x)
    assert out.shape == (2, 128, 4, 1)


def test_extractor_output_shape():
    from models.temporal_cnn import TemporalCNNExtractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = TemporalCNNExtractor(obs_space, features_dim=256)
    x = torch.randn(4, 1, 20, 13)
    with torch.no_grad():
        out = extractor(x)
    assert out.shape == (4, 256)


def test_extractor_custom_features_dim():
    from models.temporal_cnn import TemporalCNNExtractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = TemporalCNNExtractor(obs_space, features_dim=128)
    x = torch.randn(1, 1, 20, 13)
    with torch.no_grad():
        out = extractor(x)
    assert out.shape == (1, 128)


def test_extractor_uses_elu():
    from models.temporal_cnn import TemporalCNNExtractor
    import torch.nn as nn
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = TemporalCNNExtractor(obs_space)
    has_elu = any(isinstance(m, nn.ELU) for m in extractor.modules())
    assert has_elu, "ELU activation not found in extractor"
