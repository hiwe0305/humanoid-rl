"""
humanoid_rl/envs/mujoco/curriculum.py
Curriculum Learning Manager — tự động progression qua các phase training.

Chiến lược curriculum:
  Phase 0: STAND    (2M steps, target_speed=0.0)
            Mục tiêu: đứng thẳng, CoM trên chân, không ngã
  Phase 1: SLOW     (3M steps, target_speed=0.3 m/s)
            Mục tiêu: đi chậm có gait pattern
  Phase 2: WALK     (5M steps, target_speed=1.0 m/s)
            Mục tiêu: đi chuẩn với domain randomization
  Phase 3: GAIT     (3M steps, target_speed=1.0 m/s + gait rewards)
            Mục tiêu: gait đẹp, foot clearance, alternating contact
  Phase 4: TERRAIN  (5M steps, various terrains)
            Mục tiêu: generalization qua rough/stair/slope/hurdle/gap

Mỗi phase tự động start khi phase trước đạt target_return.
"""

from __future__ import annotations
from typing import Callable
import numpy as np


DEFAULT_CURRICULUM = [
    {
        "name":        "STAND",
        "task":        "stand",
        "total_steps": 2_000_000,
        "target_speed": 0.0,
        "terrain":     "flat",
        "domain_rand": False,
        "push_rand":   False,
        "gait_rewards": False,
        "lr":          3e-4,
        "entropy_coef": 0.01,
        "target_return": 400.0,       # đạt reward này → auto advance
        "min_progress": 100,          # steps giữa các lần check advance
    },
    {
        "name":        "SLOW WALK",
        "task":        "walk",
        "total_steps": 3_000_000,
        "target_speed": 0.3,
        "terrain":     "flat",
        "domain_rand": False,
        "push_rand":   False,
        "gait_rewards": False,
        "lr":          2e-4,
        "entropy_coef": 0.01,
        "target_return": 600.0,
        "min_progress": 200,
    },
    {
        "name":        "WALK",
        "task":        "walk",
        "total_steps": 5_000_000,
        "target_speed": 1.0,
        "terrain":     "flat",
        "domain_rand": False,         # bắt đầu có domain rand ở sub-phase
        "push_rand":   False,
        "gait_rewards": False,
        "lr":          1e-4,
        "entropy_coef": 0.005,
        "target_return": 800.0,
        "min_progress": 300,
    },
    {
        "name":        "GAIT",
        "task":        "walk",
        "total_steps": 3_000_000,
        "target_speed": 1.0,
        "terrain":     "flat",
        "domain_rand": True,
        "push_rand":   True,
        "gait_rewards": True,          # BẬT gait rewards — học alternating foot pattern
        "lr":          5e-5,
        "entropy_coef": 0.003,
        "target_return": 1000.0,
        "min_progress": 300,
    },
    {
        "name":        "ROUGH TERRAIN",
        "task":        "walk",
        "total_steps": 5_000_000,
        "target_speed": 1.0,
        "terrain":     "rough",
        "domain_rand": True,
        "push_rand":   True,
        "gait_rewards": True,
        "lr":          3e-5,
        "entropy_coef": 0.002,
        "target_return": None,        # None = chạy hết steps, không auto-advance
        "min_progress": 0,
    },
]


class CurriculumManager:
    """
    Quản lý curriculum training progression.
    
    - Config-driven phases
    - Auto-advance khi đạt target_return
    - Track metrics per phase
    """
    def __init__(self, curriculum: list[dict] | None = None):
        self.phases = curriculum or DEFAULT_CURRICULUM
        self._current_idx = 0
        self._phase_step = 0
        self._phase_best_return = 0.0
        self._advance_countdown = 0
        self._history: list[dict] = []

    @property
    def current_phase(self) -> dict:
        return self.phases[self._current_idx]

    @property
    def phase_index(self) -> int:
        return self._current_idx

    @property
    def is_done(self) -> bool:
        return self._current_idx >= len(self.phases)

    def step(self, metrics: dict) -> dict | None:
        """
        Gọi sau mỗi PPO update. 
        Returns phase config dict nếu chuyển phase, None nếu tiếp tục.
        """
        self._phase_step += 1
        phase = self.current_phase

        # Track best return
        ep_return = metrics.get("mean_ep_return", 0.0)
        self._phase_best_return = max(self._phase_best_return, ep_return)

        target = phase.get("target_return")
        if target is None:
            return None  # chạy hết steps

        # Check auto-advance
        self._advance_countdown = max(0, self._advance_countdown - 1)
        if (self._advance_countdown == 0 and
            self._phase_best_return >= target and
            ep_return >= target * 0.9):
            # Stay in current phase a bit longer to stabilize
            self._advance_countdown = phase.get("min_progress", 100)
            if self._phase_step >= self._advance_countdown:
                return self._advance()

        # Time-based advance (if stuck too long)
        if self._phase_step >= int(phase["total_steps"] / (256 * 16)):  # approx updates
            return self._advance()

        return None

    def _advance(self) -> dict:
        """Move to next phase."""
        self._history.append({
            "phase": self.current_phase["name"],
            "steps": self._phase_step,
            "best_return": self._phase_best_return,
        })
        old_idx = self._current_idx
        self._current_idx += 1
        self._phase_step = 0
        self._phase_best_return = 0.0
        self._advance_countdown = 0

        if self.is_done:
            return None  # training complete

        print(f"\n{'='*60}")
        print(f"  CURRICULUM ADVANCE: Phase {old_idx} → {self._current_idx}")
        print(f"  {self.phases[old_idx]['name']} → {self.current_phase['name']}")
        print(f"{'='*60}\n")
        return self.current_phase

    def reset_phase(self) -> None:
        """Reset phase tracking (gọi khi resume training)."""
        self._phase_step = 0
        self._phase_best_return = 0.0
        self._advance_countdown = 0

    def get_summary(self) -> str:
        lines = ["Curriculum Summary:"]
        for i, p in enumerate(self.phases):
            marker = ">>>" if i == self._current_idx else "   "
            done = "✓" if any(h["phase"] == p["name"] for h in self._history) else " "
            lines.append(f"  {marker} [{done}] Phase {i}: {p['name']}")
        return "\n".join(lines)
