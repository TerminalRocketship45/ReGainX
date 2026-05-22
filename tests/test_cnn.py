import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import pytest


def test_extractor_output_shape():
    from models.cnn_lstm import CNNLSTMExtractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = CNNLSTMExtractor(obs_space, features_dim=256)
    x = torch.randn(4, 1, 20, 13)
    with torch.no_grad():
        out = extractor(x)
    assert out.shape == (4, 256)


def test_extractor_custom_features_dim():
    from models.cnn_lstm import CNNLSTMExtractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = CNNLSTMExtractor(obs_space, features_dim=128)
    x = torch.randn(1, 1, 20, 13)
    with torch.no_grad():
        out = extractor(x)
    assert out.shape == (1, 128)


def test_extractor_uses_elu():
    from models.cnn_lstm import CNNLSTMExtractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = CNNLSTMExtractor(obs_space)
    has_elu = any(isinstance(m, nn.ELU) for m in extractor.modules())
    assert has_elu, "ELU activation not found in extractor"


def test_extractor_has_lstm():
    from models.cnn_lstm import CNNLSTMExtractor
    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                               shape=(1, 20, 13), dtype=np.float32)
    extractor = CNNLSTMExtractor(obs_space)
    has_lstm = any(isinstance(m, nn.LSTM) for m in extractor.modules())
    assert has_lstm, "LSTM module not found in extractor"
