"""
humanoid_rl/envs/mujoco/tasks/jump_task.py
Jump task: clear hurdle obstacles by jumping.
Terrain: hurdles at 2.5m intervals, 25cm tall.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class JumpTask:
    target_speed: float = 1.5         # moderate forward speed
    max_steps:    int   = 1500
    terrain:      str   = "hurdle"

    # Terrain config
    n_hurdles:      int   = 4
    hurdle_height:  float = 0.25       # m
    spacing:        float = 2.5        # m between hurdles

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  1.5,
            "lateral_vel_penalty": 0.5,
            "height_gain_weight":  1.0,
            "vertical_vel_weight": 2.0,    # ★ reward upward velocity (jump-specific)
        },
        "stability": {
            "upright_weight":   1.0,
            "alive_bonus":      1.5,
            "min_torso_height": 0.5,
        },
        "energy": {
            "torque_penalty":      0.0002,  # allow high torque for jumping
            "action_rate_penalty": 0.0002,
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed":   self.target_speed,
            "max_steps":      self.max_steps,
            "terrain":        self.terrain,
            "terrain_kwargs": {
                "n_hurdles":     self.n_hurdles,
                "hurdle_height": self.hurdle_height,
                "spacing":       self.spacing,
            },
            "rewards": self.rewards,
        }
