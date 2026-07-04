"""
humanoid_rl/rewards/stability.py
Stability + balance reward functions.

Includes:
    - alive bonus (height check)
    - upright reward (torso z-axis alignment)
    - CoM over feet (center of mass projection on support polygon)
    - CoM velocity penalty
    - Orientation reward (roll/pitch minimization)
    - Angular velocity penalty (reduce oscillation)
    - Foot contact reward (both feet on ground)
"""

from __future__ import annotations

import numpy as np


class StabilityRewards:
    """
    Reward for maintaining stable, upright, balanced posture.

    Terms:
        alive:         constant reward while standing
        upright:       torso z-axis aligned with world up
        com_over_feet: CoM projected inside base of support
        com_vel:       penalize horizontal CoM velocity
        orientation:   minimize roll & pitch
        angular_vel:   penalize roll/pitch rates
        foot_contact:  reward both feet touching ground
    """

    def __init__(self, cfg: dict):
        self.upright_weight       = cfg.get("upright_weight",       1.0)
        self.alive_bonus          = cfg.get("alive_bonus",          1.0)
        self.min_torso_height     = cfg.get("min_torso_height",     0.8)
        # Balance-specific
        self.com_over_feet_weight = cfg.get("com_over_feet_weight", 0.0)
        self.com_vel_penalty      = cfg.get("com_vel_penalty",      0.0)
        self.orientation_weight   = cfg.get("orientation_weight",   0.0)
        self.angular_vel_penalty  = cfg.get("angular_vel_penalty",  0.0)
        self.foot_contact_weight  = cfg.get("foot_contact_weight",  0.0)

    def compute(self, state: dict) -> float:
        reward = 0.0

        # ── Alive bonus ──────────────────────────────────────────
        torso_h = state["torso_height"]
        if torso_h > self.min_torso_height:
            reward += self.alive_bonus

        # ── Upright reward ───────────────────────────────────────
        upright_vec = state.get("upright_vec", np.array([0, 0, 1]))
        upright_score = float(np.clip(upright_vec[2], -1.0, 1.0))
        reward += self.upright_weight * upright_score

        # ── CoM over feet ────────────────────────────────────────
        if self.com_over_feet_weight > 0:
            com_rel = state.get("com_rel", np.zeros(2))
            # exp(-k * ||com_rel||²) → 1.0 when centered, ~0 when off
            com_dist_sq = float(com_rel[0]**2 + com_rel[1]**2)
            reward += self.com_over_feet_weight * np.exp(-100.0 * com_dist_sq)

        # ── CoM velocity penalty ─────────────────────────────────
        if self.com_vel_penalty > 0:
            com_vel = state.get("com_vel_xy", np.zeros(2))
            reward -= self.com_vel_penalty * float(com_vel[0]**2 + com_vel[1]**2)

        # ── Orientation reward ───────────────────────────────────
        if self.orientation_weight > 0:
            roll  = state.get("roll",  0.0)
            pitch = state.get("pitch", 0.0)
            reward += self.orientation_weight * np.exp(-10.0 * (roll**2 + pitch**2))

        # ── Angular velocity penalty ─────────────────────────────
        if self.angular_vel_penalty > 0:
            roll_rate  = state.get("roll_rate",  0.0)
            pitch_rate = state.get("pitch_rate", 0.0)
            reward -= self.angular_vel_penalty * (roll_rate**2 + pitch_rate**2)

        # ── Foot contact reward ──────────────────────────────────
        if self.foot_contact_weight > 0:
            contacts = state.get("foot_contacts", np.zeros(2))
            # Reward both feet on ground: +1 for each
            reward += self.foot_contact_weight * float(np.sum(contacts))

        return reward

    def reset(self):
        """Reset internal state (called on env reset)."""
        pass
