"""
humanoid_rl/rewards/energy.py
Energy efficiency rewards: torque penalty, action rate penalty.
Encourages smooth, energy-efficient motions.
"""

from __future__ import annotations

import numpy as np


class EnergyRewards:
    """
    Penalize high energy consumption for smoother motion.

    Terms:
        torque_penalty:       penalize large actuator torques
        action_rate_penalty:  penalize large changes in action (jerk)
    """

    def __init__(self, cfg: dict):
        self.torque_penalty      = cfg.get("torque_penalty",       0.001)
        self.action_rate_penalty = cfg.get("action_rate_penalty",  0.0005)
        self._prev_action: np.ndarray | None = None

    def compute(self, state: dict) -> float:
        reward = 0.0
        torques = state.get("torques", np.zeros(1))
        action  = state.get("action",  np.zeros(1))

        # Torque penalty (L2 norm squared)
        reward -= self.torque_penalty * float(np.sum(torques ** 2))

        # Action rate penalty (smoothness)
        if self._prev_action is not None and self.action_rate_penalty > 0:
            delta = action - self._prev_action
            reward -= self.action_rate_penalty * float(np.sum(delta ** 2))

        self._prev_action = action.copy()
        return reward

    def reset(self):
        self._prev_action = None
