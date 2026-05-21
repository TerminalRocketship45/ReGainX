import numpy as np
import pytest

myosuite = pytest.importorskip("myosuite")


@pytest.fixture
def healthy_policy_path(tmp_path):
    """Train a minimal healthy policy for wrapper tests."""
    from myosuite.utils import gym
    from stable_baselines3 import PPO
    env = gym.make("myoElbowPose1D6MRandom-v0")
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=1000)
    path = str(tmp_path / "healthy_test.zip")
    model.save(path)
    env.close()
    return path


@pytest.fixture
def exo_env(healthy_policy_path):
    import myosuite
    from myosuite.utils import gym
    from envs.elbow_env import CombinedExoOnlyWrapper
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base, frozen_policy_path=healthy_policy_path,
        bradykinesia=False, smart_reset=True, hide_pose_err=True,
    )
    yield env
    env.close()


def test_observation_space_shape(exo_env):
    obs, _ = exo_env.reset()
    assert obs.shape == exo_env.observation_space.shape
    assert len(obs.shape) == 1


def test_action_space_shape(exo_env):
    assert exo_env.action_space.shape == (1,)
    assert exo_env.action_space.low[0] == pytest.approx(0.0)
    assert exo_env.action_space.high[0] == pytest.approx(1.0)


def test_step_returns_correct_obs_shape(exo_env):
    obs, _ = exo_env.reset()
    action = exo_env.action_space.sample()
    next_obs, reward, done, truncated, info = exo_env.step(action)
    assert next_obs.shape == obs.shape
    assert isinstance(reward, float)


def test_bradykinesia_modifies_gear(healthy_policy_path):
    import myosuite
    from myosuite.utils import gym
    from envs.elbow_env import CombinedExoOnlyWrapper
    base = gym.make("myoFatiElbowPose1D6MExoRandom-v0")
    env = CombinedExoOnlyWrapper(
        base, frozen_policy_path=healthy_policy_path,
        bradykinesia=True, smart_reset=True, hide_pose_err=True,
    )
    env.reset()
    assert env.force_scale < 1.0
    assert env.activation_slowdown > 1.0
    env.close()


def test_get_combined_info_returns_dict(exo_env):
    exo_env.reset()
    info = exo_env.get_combined_info()
    assert "fatigue" in info
    assert "bradykinesia" in info
    assert "avg_mf" in info["fatigue"]
    assert "force_scale" in info["bradykinesia"]
    assert "activation_slowdown" in info["bradykinesia"]
