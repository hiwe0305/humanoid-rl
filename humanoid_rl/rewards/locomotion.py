"""
humanoid_rl/rewards/locomotion.py
Locomotion reward functions: forward velocity, lateral penalty, height gain.
"""

from __future__ import annotations

import numpy as np


class LocomotionRewards:
    """
    Reward for forward locomotion behavior.

    Terms:
        forward_vel:   reward matching target speed in x-direction
        lateral_vel:   penalize lateral (y) drift
        height_gain:   reward gaining height (stair/jump tasks)
        vertical_vel:  reward upward velocity (jump tasks)
    """

    def __init__(self, cfg: dict):
        self.forward_vel_weight   = cfg.get("forward_vel_weight",   2.0)
        self.lateral_vel_penalty  = cfg.get("lateral_vel_penalty",  0.5)
        self.height_gain_weight   = cfg.get("height_gain_weight",   0.0)
        self.vertical_vel_weight  = cfg.get("vertical_vel_weight",  0.0)
        self._prev_torso_height: float | None = None

    def compute(self, state: dict) -> float:
        reward = 0.0
        root_vel    = state["root_vel"]      # [vx, vy, vz]
        target_spd  = state["target_speed"]

        # Forward velocity reward: exponential kernel (smooth, bounded)
        vx = root_vel[0]
        vel_error  = abs(vx - target_spd)
        reward += self.forward_vel_weight * np.exp(-vel_error ** 2)

        # Lateral velocity penalty
        vy = root_vel[1]
        reward -= self.lateral_vel_penalty * vy ** 2

        # Height gain (stair/jump)
        if self.height_gain_weight > 0:
            curr_h = state["torso_height"]
            if self._prev_torso_height is not None:
                dh = curr_h - self._prev_torso_height
                reward += self.height_gain_weight * max(dh, 0.0)
            self._prev_torso_height = curr_h

        # Vertical velocity (jump task)
        if self.vertical_vel_weight > 0:
            vz = root_vel[2]
            reward += self.vertical_vel_weight * max(vz, 0.0)

        return reward

    def reset(self):
        self._prev_torso_height = None
