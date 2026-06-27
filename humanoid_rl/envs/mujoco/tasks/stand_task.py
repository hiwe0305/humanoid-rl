"""
humanoid_rl/envs/mujoco/tasks/stand_task.py
Phase 0: Học đứng vững (không di chuyển).

Mục tiêu: robot đứng thẳng, torso ổn định, không ngã.
Không có yêu cầu forward velocity — chỉ cần đứng.

Reward:
  upright    = 4.0 × cos(torso tilt)   ← chính: đứng thẳng
  alive      = 1.0 per step
  still      = 1.0 × exp(-vx² - vy²)  ← thưởng đứng yên
  energy     = -0.0001 × Σ τ²          ← tiết kiệm năng lượng
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StandTask:
    max_steps: int = 1000   # dài hơn: buộc robot học cân bằng lâu dài
    terrain:   str = "flat"

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  1.0,    # nhỏ: thưởng nhẹ nếu vx ≈ 0
            "backward_penalty":    3.0,    # phạt đi lùi
            "lateral_vel_penalty": 1.0,    # phạt trượt ngang
            "height_gain_weight":  0.0,
            "vertical_vel_weight": 0.0,
        },
        "stability": {
            "upright_weight":   4.0,   # mạnh nhất: đứng thẳng
            "alive_bonus":      1.0,
            "min_torso_height": 0.5,   # cao hơn: terminate sớm nếu đổ
        },
        "energy": {
            "torque_penalty":      0.0001,
            "action_rate_penalty": 0.005,
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed": 0.0,     # đứng yên
            "max_steps":    self.max_steps,
            "terrain":      self.terrain,
            "rewards":      self.rewards,
        }
