"""
humanoid_rl/envs/mujoco/tasks/walk_task.py
Walk task với gait rewards (CPG tracking + foot contact + clearance).
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class WalkTask:
    target_speed: float = 1.0
    max_steps:    int   = 1000
    terrain:      str   = "flat"

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  4.0,
            "backward_penalty":    5.0,
            "lateral_vel_penalty": 0.3,
            "height_gain_weight":  0.0,
            "vertical_vel_weight": 0.0,
        },
        "stability": {
            "upright_weight":   2.0,    # giảm một chút — gait reward đã xử lý balance
            "alive_bonus":      0.5,
            "min_torso_height": 0.3,
        },
        "energy": {
            "torque_penalty":      0.0001,
            "action_rate_penalty": 0.005,
        },
        "gait": {
            "cpg_weight":         2.0,   # tăng: buộc bám CPG reference
            "contact_weight":     3.0,   # tăng: alternating contact
            "clearance_weight":   2.0,   # tăng: nhấc chân
            "symmetry_weight":    0.3,
            "foot_slip_weight":   1.0,   # tăng: không trượt
            "double_stance_pen":  4.0,   # NEW: phạt cả 2 chân cùng tiếp đất
            "clearance_height":   0.08,
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed": self.target_speed,
            "max_steps":    self.max_steps,
            "terrain":      self.terrain,
            "rewards":      self.rewards,
        }
