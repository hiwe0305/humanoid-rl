"""
humanoid_rl/envs/mujoco/tasks/walk_task.py
Walk task: maintain forward velocity 1.0 m/s on flat or rough terrain.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class WalkTask:
    target_speed: float = 1.0   # m/s
    max_steps:    int   = 1000
    terrain:      str   = "flat"   # "flat" | "rough"

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  2.0,
            "lateral_vel_penalty": 0.5,
            "height_gain_weight":  0.0,
            "vertical_vel_weight": 0.0,
        },
        "stability": {
            "upright_weight":   1.5,
            "alive_bonus":      1.0,
            "min_torso_height": 0.5,
        },
        "energy": {
            "torque_penalty":      0.0005,
            "action_rate_penalty": 0.001,
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed": self.target_speed,
            "max_steps":    self.max_steps,
            "terrain":      self.terrain,
            "rewards":      self.rewards,
        }
