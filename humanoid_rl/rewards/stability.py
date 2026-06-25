"""
humanoid_rl/rewards/stability.py
Stability reward functions: upright posture, minimum height (alive bonus).
"""

from __future__ import annotations

import numpy as np


class StabilityRewards:
    """
    Reward for maintaining stable, upright posture.

    Terms:
        alive:     constant reward while robot is upright and above min height
        upright:   reward for torso z-axis alignment with world up
    """

    def __init__(self, cfg: dict):
        self.upright_weight  = cfg.get("upright_weight",  1.0)
        self.alive_bonus     = cfg.get("alive_bonus",     1.0)
        self.min_torso_height = cfg.get("min_torso_height", 0.8)

    def compute(self, state: dict) -> float:
        reward = 0.0

        # Alive bonus (simple height check)
        torso_h = state["torso_height"]
        if torso_h > self.min_torso_height:
            reward += self.alive_bonus

        # Upright reward: dot product of torso up-vector with world up [0,0,1]
        upright_vec = state.get("upright_vec", np.array([0, 0, 1]))
        upright_score = float(np.clip(upright_vec[2], -1.0, 1.0))  # z-component
        reward += self.upright_weight * upright_score

        return reward
