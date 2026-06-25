"""
humanoid_rl/envs/mujoco/tasks/stair_task.py
Stair task: ascend a staircase — requires lifting feet and gaining height.
Terrain: stair (6 steps, 15cm each).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StairTask:
    target_speed: float = 0.5         # slow forward progress up stairs
    max_steps:    int   = 2000        # more time to climb
    terrain:      str   = "stair"

    # Terrain configuration
    n_steps:      int   = 6
    step_height:  float = 0.15       # m
    step_depth:   float = 0.40       # m

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  1.5,
            "lateral_vel_penalty": 1.0,
            "height_gain_weight":  3.0,    # ★ reward gaining height (stair-specific)
            "vertical_vel_weight": 0.5,    # reward upward velocity
        },
        "stability": {
            "upright_weight":   2.0,       # upright posture more important
            "alive_bonus":      1.5,
            "min_torso_height": 0.5,       # relative to floor
        },
        "energy": {
            "torque_penalty":      0.001,
            "action_rate_penalty": 0.001,
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed":    self.target_speed,
            "max_steps":       self.max_steps,
            "terrain":         self.terrain,
            "terrain_kwargs":  {
                "n_steps":     self.n_steps,
                "step_height": self.step_height,
                "step_depth":  self.step_depth,
            },
            "rewards":         self.rewards,
        }
