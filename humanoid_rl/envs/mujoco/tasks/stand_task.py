"""
humanoid_rl/envs/mujoco/tasks/stand_task.py
Phase 0: Học đứng cân bằng (balance).

Mục tiêu: robot tự giữ thăng bằng — CoM nằm trên chân, body thẳng, không ngã.
Đây là FOUNDATION cho tất cả 4 tasks khác (walk, run, jump).

Reward (balance-focused):
  com_over_feet  = 5.0 × exp(-100 × ||com_rel||²)  ← CHÍNH: CoM trên chân
  orientation    = 3.0 × exp(-10 × (roll² + pitch²)) ← đứng thẳng
  alive          = 2.0 per step                       ← sống lâu
  upright        = 3.0 × cos(torso tilt)              ← torso thẳng
  foot_contact   = 1.0 × (left + right contact)       ← 2 chân chạm đất
  com_vel        = -2.0 × (com_vx² + com_vy²)         ← đứng yên
  angular_vel    = -1.0 × (roll_rate² + pitch_rate²)   ← không lắc
  energy         = -0.0005 × Σ τ²                      ← tiết kiệm năng lượng
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class StandTask:
    max_steps: int = 500    # ngắn hơn → reset nhanh hơn → learning hiệu quả
    terrain:   str = "flat"

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {
            "forward_vel_weight":  0.5,    # nhỏ: chỉ nhẹ thưởng vx ≈ 0
            "backward_penalty":    1.0,    # phạt đi lùi
            "lateral_vel_penalty": 1.0,    # phạt trượt ngang
            "height_gain_weight":  0.0,
            "vertical_vel_weight": 0.0,
        },
        "stability": {
            "upright_weight":       3.0,   # đứng thẳng
            "alive_bonus":          2.0,   # sống lâu → thưởng nhiều
            "min_torso_height":     0.5,   # cho robot cơ hội phục hồi
            # Balance-specific rewards
            "com_over_feet_weight": 5.0,   # MẠNH NHẤT: CoM nằm trên chân
            "com_vel_penalty":      2.0,   # phạt CoM di chuyển nhanh
            "orientation_weight":   3.0,   # thưởng body thẳng (roll/pitch nhỏ)
            "angular_vel_penalty":  1.0,   # phạt lắc lư
            "foot_contact_weight":  1.0,   # thưởng 2 chân chạm đất
        },
        "energy": {
            "torque_penalty":      0.0005, # phạt mạnh hơn → tiết kiệm
            "action_rate_penalty": 0.01,   # phạt thay đổi action nhanh
        },
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed": 0.0,     # đứng yên
            "max_steps":    self.max_steps,
            "terrain":      self.terrain,
            "rewards":      self.rewards,
        }
