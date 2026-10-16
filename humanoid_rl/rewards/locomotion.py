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
        forward_vel:    reward matching target speed in x-direction
        backward_pen:   STRONG penalty for backward motion (vx < 0)
        lateral_vel:    penalize lateral (y) drift
        height_gain:    reward gaining height (stair/jump tasks)
        vertical_vel:   reward upward velocity (jump tasks)
    """

    def __init__(self, cfg: dict):
        self.forward_vel_weight   = cfg.get("forward_vel_weight",   2.0)
        self.backward_penalty     = cfg.get("backward_penalty",     0.0)   # NEW
        self.lateral_vel_penalty  = cfg.get("lateral_vel_penalty",  0.5)
        self.height_gain_weight   = cfg.get("height_gain_weight",   0.0)
        self.vertical_vel_weight  = cfg.get("vertical_vel_weight",  0.0)
        self._prev_torso_height: float | None = None

    def compute(self, state: dict) -> float:
        reward = 0.0
        root_vel    = state["root_vel"]      # [vx, vy, vz]
        target_spd  = state["target_speed"]

        # Forward velocity reward: exponential kernel (smooth, bounded).
        # Kernel width grows with target speed so there is still a learning
        # signal when starting from standing still (exp(-(0-3)^2)=1e-4 otherwise).
        vx = root_vel[0]
        vel_error  = vx - target_spd
        sigma      = max(1.0, target_spd / 1.5)
        reward += self.forward_vel_weight * np.exp(-(vel_error / sigma) ** 2)

        # Backward motion penalty (CRITICAL for walk training)
        # If vx < 0: penalty = backward_penalty × vx²
        # Provides STRONG gradient signal to avoid walking backward
        if self.backward_penalty > 0 and vx < 0:
            reward -= self.backward_penalty * vx ** 2

        # Lateral velocity penalty
        vy = root_vel[1]
        reward -= self.lateral_vel_penalty * vy ** 2

        # Height gain (stair/jump): reward only NEW record heights within the
        # episode. Per-step max(dh,0) can be farmed by bobbing up and down.
        if self.height_gain_weight > 0:
            curr_h = state["torso_height"]
            if self._prev_torso_height is None:
                self._prev_torso_height = curr_h   # running max height
            elif curr_h > self._prev_torso_height:
                reward += self.height_gain_weight * (curr_h - self._prev_torso_height)
                self._prev_torso_height = curr_h

        # Vertical velocity (jump task)
        if self.vertical_vel_weight > 0:
            vz = root_vel[2]
            reward += self.vertical_vel_weight * max(vz, 0.0)

        return reward

    def reset(self):
        self._prev_torso_height = None
