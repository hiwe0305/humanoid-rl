"""
humanoid_rl/envs/mujoco/tasks/run_task.py
Run task: maintain 3.0 m/s forward velocity — higher speed than walk.
Terrain: rough blocks to handle uneven ground while running.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RunTask:
    target_speed: float = 3.0
    max_steps:    int   = 1000
    terrain:      str   = "rough"

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  3.0,    # higher weight for speed
            "lateral_vel_penalty": 1.0,    # penalize lateral drift more
            "height_gain_weight":  0.0,
            "vertical_vel_weight": 0.0,
        },
        "stability": {
            "upright_weight":   1.0,
            "alive_bonus":      1.0,
            "min_torso_height": 0.5,
        },
        "energy": {
            "torque_penalty":      0.0002, # less penalty → allow more torque for speed
            "action_rate_penalty": 0.0005,
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed": self.target_speed,
            "max_steps":    self.max_steps,
            "terrain":      self.terrain,
            "rewards":      self.rewards,
        }
