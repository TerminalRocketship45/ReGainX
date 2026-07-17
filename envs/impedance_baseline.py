"""
Velocity-deficit impedance controller for ReGainX comparison.

Implements the assist-as-needed (AAN) paradigm described in:
  Dalla Gasperina et al. (2021). Review on patient-cooperative control
  strategies for upper-limb rehabilitation exoskeletons.
  Frontiers in Robotics and AI. PMC8688994.

The controller applies assistive torque proportional to the velocity
deficit (difference between current velocity and a healthy baseline),
in the direction of ongoing movement. It requires no calibration and no
training, modelling the simplest form of patient-cooperative impedance
control: pure assist-as-needed.
"""

import numpy as np


class ImpedanceBaseline:
    """
    Velocity-deficit PD impedance controller (non-learning, no calibration).

    At each timestep:
    - Reads joint velocity from the last element of the observation.
    - Estimates the velocity deficit: max(0, v_healthy - |dq|).
    - Applies torque proportional to deficit in the direction of recent motion.
    - A rolling window of recent velocities is used to estimate movement intent.

    Parameters
    ----------
    healthy_velocity : float
        Target mean joint velocity (rad/s) representing healthy elbow movement.
        0.8 rad/s is a conservative estimate for typical functional reach speed.
    kp : float
        Proportional gain on velocity deficit.
    kd : float
        Derivative gain on velocity change (reduces oscillation).
    direction_window : int
        Number of steps to average for motion-direction estimation.
    """

    def __init__(
        self,
        healthy_velocity: float = 0.8,
        kp: float = 0.6,
        kd: float = 0.05,
        direction_window: int = 5,
    ):
        self.healthy_velocity = healthy_velocity
        self.kp = kp
        self.kd = kd
        self.direction_window = direction_window
        self.reset()

    def reset(self) -> None:
        self._vel_history: list = []
        self._prev_vel: float = 0.0

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute assistive torque.

        Parameters
        ----------
        obs : np.ndarray, shape (8,)
            [act_0, ..., act_5, q, dq]  — joint velocity is the last element.

        Returns
        -------
        np.ndarray, shape (1,)  in [0, 1]
        """
        dq = float(obs[-1])

        self._vel_history.append(dq)
        if len(self._vel_history) > self.direction_window:
            self._vel_history.pop(0)

        mean_vel = float(np.mean(self._vel_history))
        if abs(mean_vel) < 1e-4:
            self._prev_vel = dq
            return np.array([0.0], dtype=np.float32)

        direction = float(np.sign(mean_vel))
        deficit   = max(0.0, self.healthy_velocity - abs(dq))
        vel_diff  = abs(dq - self._prev_vel)
        self._prev_vel = dq

        torque = direction * (self.kp * deficit + self.kd * vel_diff)
        torque = float(np.clip(torque, 0.0, 1.0))
        return np.array([torque], dtype=np.float32)
