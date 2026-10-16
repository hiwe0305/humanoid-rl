"""
humanoid_rl/rewards/gait.py
Gait quality rewards — v2: fix degenerate "sliding-forward" gait.

Root cause của sliding: cả 2 chân luôn chạm sàn → penalty không đủ mạnh.

Fixes:
  1. PENALIZE both-feet-on-ground STRONGLY (phá local optima trượt tới)
  2. REWARD foot lift independently (không cần alternating, chỉ cần nhấc chân)
  3. CPG tracking với larger weight để force gait pattern
  4. Foot velocity penalty trên cả 2 chân khi cùng tiếp đất
"""
from __future__ import annotations
import numpy as np

# H1 joint indices (trong qpos[7:])
L_HIP_PITCH = 2;  R_HIP_PITCH = 7
L_KNEE      = 3;  R_KNEE      = 8
L_ANKLE     = 4;  R_ANKLE     = 9
L_HIP_ROLL  = 1;  R_HIP_ROLL  = 6

# Default gait period (seconds). If control_dt is known, GaitRewards
# converts this to steps internally. The period is set via config or
# defaults to 0.8s (matching H1 walking gait at ~1.0 m/s).
DEFAULT_GAIT_PERIOD_S = 0.8
HIP_AMP    = 0.20   # < H1_ACTION_SCALE (0.3) so the reference stays reachable
KNEE_AMP   = 0.25
ANKLE_AMP  = 0.12
HIP_ROLL_AMP = 0.08
FOOT_TARGET_HEIGHT = 0.08  # m — target foot clearance


class GaitRewards:
    """
    v2: Fix degenerate sliding gait.
    Key insight: phải phạt nặng khi CẢ 2 chân tiếp đất (double stance > 50% thời gian).
    """

    def __init__(self, cfg: dict, control_dt: float = 0.025, default_q: np.ndarray | None = None):
        self.cpg_weight         = cfg.get("cpg_weight",         2.0)   # tăng
        self.contact_weight     = cfg.get("contact_weight",     3.0)   # tăng
        self.clearance_weight   = cfg.get("clearance_weight",   2.0)   # tăng
        self.symmetry_weight    = cfg.get("symmetry_weight",    0.3)
        self.foot_slip_weight   = cfg.get("foot_slip_weight",   1.0)   # tăng
        self.double_stance_pen  = cfg.get("double_stance_pen",  4.0)   # NEW: phạt cả 2 chân xuống
        self.clearance_height   = cfg.get("clearance_height",   FOOT_TARGET_HEIGHT)
        # Gait period in steps = period_seconds / control_dt
        gait_period_s = cfg.get("gait_period_s", DEFAULT_GAIT_PERIOD_S)
        self._gait_period_steps = max(2, int(round(gait_period_s / control_dt)))

        # Robot default (standing) joint pose. The CPG reference is defined as
        # offsets around it so it is always reachable by the PD target
        # (default +- action_scale). If None, fall back to legacy absolute pose.
        self._default_q = None if default_q is None else np.asarray(default_q, dtype=np.float32)

        self._phase: float = 0.0

    def reset(self) -> None:
        self._phase = 0.0

    def phase_features(self, step_count: int) -> np.ndarray:
        """[sin, cos] of the gait phase for control step `step_count`."""
        phi = 2 * np.pi * ((step_count / self._gait_period_steps) % 1.0)
        return np.array([np.sin(phi), np.cos(phi)], dtype=np.float32)

    def _cpg_reference(self) -> np.ndarray:
        phi = self._phase * 2 * np.pi
        if self._default_q is not None:
            q_ref = self._default_q.copy()
            hip_base = knee_base = ankle_base = 0.0   # offsets added to default below
        else:
            q_ref = np.zeros(19, dtype=np.float32)
            hip_base, knee_base, ankle_base = -0.35, 0.70, -0.35

        # Hip pitch: alternating ±HIP_AMP around default
        q_ref[L_HIP_PITCH] += hip_base + HIP_AMP * np.sin(phi)
        q_ref[R_HIP_PITCH] += hip_base - HIP_AMP * np.sin(phi)

        # Knee: bend only during swing phase
        q_ref[L_KNEE] += knee_base + KNEE_AMP * max(0.0, -np.sin(phi))
        q_ref[R_KNEE] += knee_base + KNEE_AMP * max(0.0,  np.sin(phi))

        # Ankle
        q_ref[L_ANKLE] += ankle_base - ANKLE_AMP * np.sin(phi)
        q_ref[R_ANKLE] += ankle_base + ANKLE_AMP * np.sin(phi)

        # Hip roll: weight shift
        q_ref[L_HIP_ROLL] += HIP_ROLL_AMP * np.sin(phi)
        q_ref[R_HIP_ROLL] += HIP_ROLL_AMP * np.sin(phi + np.pi)

        return q_ref

    def _expected_contacts(self) -> tuple[bool, bool]:
        left_stance  = self._phase < 0.5
        right_stance = not left_stance
        return left_stance, right_stance

    def compute(self, state: dict) -> float:
        reward     = 0.0
        step_count = state.get("step_count", 0)
        vx         = state.get("vx", 0.0)

        # Advance phase
        self._phase = (step_count / self._gait_period_steps) % 1.0

        q_joints = state["q_joints"]
        q_ref    = self._cpg_reference()
        exp_l, exp_r = self._expected_contacts()

        lc = state.get("lfoot_contact", False)
        rc = state.get("rfoot_contact", False)
        lz = state.get("lfoot_z", 0.0)
        rz = state.get("rfoot_z", 0.0)

        # ── 1. CPG Tracking ──────────────────────────────────────────────────
        leg_mask = [L_HIP_PITCH, R_HIP_PITCH, L_KNEE, R_KNEE,
                    L_ANKLE, R_ANKLE, L_HIP_ROLL, R_HIP_ROLL]
        q_err   = q_joints[leg_mask] - q_ref[leg_mask]
        cpg_err = float(np.sum(q_err ** 2))
        reward -= self.cpg_weight * cpg_err

        # ── 2. Double-stance penalty (KEY FIX) ───────────────────────────────
        # Khi cả 2 chân xuống: robot đang trượt, không phải đi bộ
        # Phạt nặng để buộc robot nhấc 1 chân lên
        if lc and rc and vx > 0.1:
            reward -= self.double_stance_pen

        # ── 3. Foot Contact Pattern ──────────────────────────────────────────
        if vx > 0.1:
            contact_match = int(lc == exp_l) + int(rc == exp_r)
            reward += self.contact_weight * contact_match / 2.0

        # ── 4. Foot Clearance (độc lập với alternating) ──────────────────────
        # Thưởng khi bất kỳ chân nào được nhấc lên
        l_swing = not exp_l
        r_swing = not exp_r

        if l_swing and lz > 0.01:
            reward += self.clearance_weight * min(lz / self.clearance_height, 1.0)
        if r_swing and rz > 0.01:
            reward += self.clearance_weight * min(rz / self.clearance_height, 1.0)

        # Phạt thêm nếu swing foot vẫn tiếp đất
        if l_swing and lc:
            reward -= self.clearance_weight * 0.5
        if r_swing and rc:
            reward -= self.clearance_weight * 0.5

        # ── 5. Foot Slip: tất cả chân tiếp đất không được trượt ─────────────
        if lc:
            lv = state.get("lfoot_vel", np.zeros(3))
            reward -= self.foot_slip_weight * float(lv[0]**2 + lv[1]**2)
        if rc:
            rv = state.get("rfoot_vel", np.zeros(3))
            reward -= self.foot_slip_weight * float(rv[0]**2 + rv[1]**2)

        # ── 6. Gait Symmetry ─────────────────────────────────────────────────
        sym_err = (
            (q_joints[L_HIP_PITCH] + q_joints[R_HIP_PITCH]) ** 2 +
            (q_joints[L_KNEE]      - q_joints[R_KNEE])      ** 2 +
            (q_joints[L_ANKLE]     + q_joints[R_ANKLE])     ** 2
        )
        reward -= self.symmetry_weight * sym_err

        return float(reward)
