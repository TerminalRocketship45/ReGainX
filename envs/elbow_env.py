"""
CombinedExoOnlyWrapper — MARL elbow exoskeleton environment wrapper.

A frozen healthy PPO policy drives the 6 muscles.
An exo policy controls one torque actuator in [0, 1].
Optionally applies bradykinesia (force reduction + activation slowdown).
Always applies muscular degeneration via smart_reset MF initialisation.

Reference: ReGainX/exoskeletons/elbow/train.ipynb cells 12-13.
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


class CombinedExoOnlyWrapper(gym.Env):
    """
    Wrap ``myoFatiElbowPose1D6MExoRandom-v0`` for exo-only RL.

    The frozen healthy policy supplies the 6-muscle actions; the learnable
    policy controls a single exo torque normalised to [0, 1].

    Parameters
    ----------
    base_env:
        An instantiated ``myoFatiElbowPose1D6MExoRandom-v0`` gym environment.
    frozen_policy_path:
        Path (or path stem) to a Stable-Baselines3 PPO ``.zip`` file trained
        on ``myoElbowPose1D6MRandom-v0``.
    bradykinesia:
        If True, randomise force_scale and activation_slowdown each episode to
        simulate Parkinson's-style motor impairment.
    smart_reset:
        If True, randomise MF/MA/MR to simulate varied fatigue levels at
        episode start.
    hide_pose_err:
        If True, strip the ``pose_err`` component from the exo observation
        (the frozen policy still receives the full observation).
    force_scale_range:
        (low, high) uniform range for bradykinesia force scaling factor.
    activation_slowdown_range:
        (low, high) uniform range for bradykinesia activation slowdown factor.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_env,
        frozen_policy_path: str,
        bradykinesia: bool = False,
        smart_reset: bool = True,
        hide_pose_err: bool = True,
        force_scale_range: tuple = (0.6, 0.9),
        activation_slowdown_range: tuple = (1.1, 1.4),
    ):
        super().__init__()

        self.base_env = base_env
        self.frozen_policy = PPO.load(frozen_policy_path)
        self.bradykinesia = bradykinesia
        self.smart_reset = smart_reset
        self.hide_pose_err = hide_pose_err
        self.force_scale_range = force_scale_range
        self.activation_slowdown_range = activation_slowdown_range

        # Bradykinesia state (randomised each reset when bradykinesia=True)
        self.force_scale: float = 1.0
        self.activation_slowdown: float = 1.0
        self._original_gear = None
        self._original_dynprm = None

        # Number of muscles from the fatigue model
        self.n_muscles = len(self.base_env.unwrapped.muscle_fatigue.MA)

        # Obs dimensions ---------------------------------------------------
        # The frozen policy was trained on the healthy env (same 9-D obs).
        self._base_obs_dim: int = self.frozen_policy.observation_space.shape[0]

        # Determine pose_err dimension from the base env obs_dict
        _pe_raw = self.base_env.unwrapped.get_obs_dict(self.base_env.unwrapped.sim).get("pose_err", np.array([]))
        _pe = np.atleast_1d(np.asarray(_pe_raw))
        self._pose_err_dim = int(_pe.size)

        # Exo obs is base obs minus pose_err (if requested)
        self._exo_obs_dim: int = (
            self._base_obs_dim - self._pose_err_dim
            if hide_pose_err
            else self._base_obs_dim
        )

        # Gymnasium spaces -------------------------------------------------
        full_space = self.base_env.observation_space

        # Exo action: single normalised torque in [0, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: sliced from the base env observation space
        self.observation_space = gym.spaces.Box(
            low=full_space.low[: self._exo_obs_dim],
            high=full_space.high[: self._exo_obs_dim],
            dtype=np.float32,
        )

        # If not using smart_reset, let the base env handle fatigue randomly
        if not smart_reset:
            self.base_env.unwrapped.set_fatigue_reset_random(True)

    # ------------------------------------------------------------------
    # Bradykinesia helpers
    # ------------------------------------------------------------------

    def _sample_brady(self) -> None:
        """Draw new bradykinesia parameters uniformly."""
        self.force_scale = float(np.random.uniform(*self.force_scale_range))
        self.activation_slowdown = float(
            np.random.uniform(*self.activation_slowdown_range)
        )

    def _apply_brady(self) -> None:
        """Scale actuator gear (force) and dynprm (activation) in the MuJoCo model."""
        model = self.base_env.unwrapped.sim.model

        # Cache originals on first application
        if self._original_gear is None:
            self._original_gear = model.actuator_gear.copy()
        model.actuator_gear[:] = self._original_gear * self.force_scale

        if hasattr(model, "actuator_dynprm") and model.actuator_dynprm.size > 0:
            if self._original_dynprm is None:
                self._original_dynprm = model.actuator_dynprm.copy()
            model.actuator_dynprm[:] = self._original_dynprm * self.activation_slowdown

    def _restore_brady(self) -> None:
        """Restore original actuator parameters before re-sampling."""
        if self._original_gear is not None:
            self.base_env.unwrapped.sim.model.actuator_gear[:] = self._original_gear
        if self._original_dynprm is not None:
            try:
                self.base_env.unwrapped.sim.model.actuator_dynprm[:] = (
                    self._original_dynprm
                )
            except AttributeError:
                self._original_dynprm = None

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _exo_obs(self, raw: np.ndarray) -> np.ndarray:
        """Return the slice of raw obs that the exo policy receives."""
        return raw[: self._exo_obs_dim].astype(np.float32)

    def _frozen_obs(self, raw: np.ndarray) -> np.ndarray:
        """Return the slice of raw obs that the frozen muscle policy receives."""
        return raw[: self._base_obs_dim].astype(np.float32)

    def _current_raw_obs(self) -> np.ndarray:
        """Fetch current observation directly from the underlying MuJoCo sim."""
        try:
            return self.base_env.unwrapped.get_obs()
        except AttributeError:
            return self.base_env._get_obs()  # fallback for older wrappers

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, **kwargs):
        """Reset episode, apply smart fatigue reset and bradykinesia if enabled."""
        # Standard reset — fatigue_reset=True triggers myosuite's built-in
        # fatigue state randomisation; we may override below via smart_reset.
        try:
            raw_obs, info = self.base_env.reset(seed=seed, fatigue_reset=True)
        except TypeError:
            raw_obs, info = self.base_env.reset(seed=seed)

        # --- Target angle randomisation ---
        tjr = self.base_env.unwrapped.target_jnt_range  # shape (n_joints, 2)
        low = tjr[:, 0]
        high = tjr[:, 1]
        self.base_env.unwrapped.target_jnt_value = np.random.uniform(low, high)
        self.base_env.unwrapped.target_type = "fixed"
        self.base_env.unwrapped.update_target(restore_sim=True)

        # --- Smart fatigue reset ---
        if self.smart_reset:
            MF = np.random.uniform(0.7, 1.0, size=self.n_muscles)
            remaining = 1.0 - MF
            split = np.random.uniform(0.0, 1.0, size=self.n_muscles)
            self.base_env.unwrapped.muscle_fatigue.MA[:] = remaining * split
            self.base_env.unwrapped.muscle_fatigue.MR[:] = remaining * (1.0 - split)
            self.base_env.unwrapped.muscle_fatigue.MF[:] = MF

        # --- Bradykinesia ---
        if self.bradykinesia:
            self._restore_brady()
            self._sample_brady()
            self._apply_brady()

        # Re-fetch obs after target + fatigue modifications
        raw_obs = self._current_raw_obs()
        return self._exo_obs(raw_obs), info

    def step(self, exo_action):
        """
        Step the environment.

        The frozen policy provides muscle actions; exo_action is prepended
        to form the full 7-D action for the base env.
        """
        exo_action = np.atleast_1d(np.asarray(exo_action, dtype=np.float32))

        # Get current raw obs for the frozen policy
        raw = self._current_raw_obs()
        muscle_actions, _ = self.frozen_policy.predict(
            self._frozen_obs(raw), deterministic=True
        )

        # Combine: [exo_torque, muscle_1, ..., muscle_6]
        full_action = np.concatenate([exo_action, muscle_actions])

        next_raw, reward, done, truncated, info = self.base_env.step(full_action)
        return self._exo_obs(next_raw), float(reward), done, truncated, info

    def render(self, *args, **kwargs):
        return self.base_env.render(*args, **kwargs)

    def close(self):
        self.base_env.close()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_combined_info(self) -> dict:
        """Return a structured dict with fatigue and bradykinesia diagnostics."""
        avg_mf = float(np.mean(self.base_env.unwrapped.muscle_fatigue.MF))
        return {
            "fatigue": {
                "avg_mf": avg_mf,
            },
            "bradykinesia": {
                "enabled": self.bradykinesia,
                "force_scale": self.force_scale,
                "activation_slowdown": self.activation_slowdown,
            },
        }
