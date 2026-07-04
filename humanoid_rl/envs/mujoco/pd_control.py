"""
humanoid_rl/envs/mujoco/pd_control.py
PD Position Controller — converts target joint positions → torques.

This is the standard control mode for real humanoid robots.
Using PD control (instead of direct torques) greatly improves sim-to-real transfer.

Control law:
    τ_i = kp_i × (q_target_i - q_i) + kd_i × (0 - dq_i)

Action space (after this layer):
    Normalized offsets from default pose [-1, 1]
    → scaled by action_scale (rad) → q_target = q_default + offset
"""
from __future__ import annotations

import numpy as np


class PDController:
    """
    Position-based PD controller for MuJoCo humanoid joints.

    Args:
        model:        mujoco.MjModel
        kp:           proportional gain (scalar or array of shape [nu])
        kd:           derivative gain   (scalar or array of shape [nu])
        torque_limit: max torque per joint (array [nu]) — clips output
        action_scale: max joint offset from default pose (radians)
    """

    def __init__(
        self,
        model,
        kp:           float | np.ndarray = 80.0,
        kd:           float | np.ndarray = 4.0,
        torque_limit: np.ndarray | None  = None,
        action_scale: float              = 0.5,
    ):
        nu = model.nu
        self.nu           = nu
        self.kp           = np.broadcast_to(np.asarray(kp, dtype=np.float32), (nu,)).copy()
        self.kd           = np.broadcast_to(np.asarray(kd, dtype=np.float32), (nu,)).copy()
        self.action_scale = float(action_scale)

        # Joint ranges from model
        self._jnt_range = np.zeros((nu, 2), dtype=np.float32)
        for i in range(nu):
            jid = model.actuator_trnid[i, 0]
            if jid >= 0 and model.jnt_limited[jid]:
                self._jnt_range[i] = model.jnt_range[jid]
            else:
                self._jnt_range[i] = [-np.pi, np.pi]

        # Default joint positions (from model initial state)
        self._q_default = np.zeros(nu, dtype=np.float32)

        # Gear ratio: motor actuator force = ctrl * gear
        # We compute torques in joint space, then divide by gear for ctrl
        self._gear = np.ones(nu, dtype=np.float32)
        if hasattr(model, 'actuator_gear') and model.actuator_gear is not None:
            gear = model.actuator_gear[:, 0].copy()
            if not np.all(gear == 0):
                self._gear = np.abs(gear).astype(np.float32)

        # Torque limits
        if torque_limit is not None:
            self.torque_limit = np.asarray(torque_limit, dtype=np.float32)
        else:
            # Get from model, fall back to large default
            tl = model.actuator_forcerange[:, 1].copy()
            self.torque_limit = tl if not np.all(tl == 0) else np.ones(nu) * 300.0

    def set_default_pose(self, qpos: np.ndarray, qpos_offset: int = 7) -> None:
        """Set default (standing) joint pose from qpos[qpos_offset:]."""
        self._q_default = qpos[qpos_offset: qpos_offset + self.nu].copy().astype(np.float32)

    def compute_torques(
        self,
        action: np.ndarray,   # normalized action [-1, 1], shape [nu]
        q: np.ndarray,        # current joint positions, shape [nq], full qpos
        dq: np.ndarray,       # current joint velocities, shape [nv], full qvel
        qpos_offset: int = 7, # index where joint positions start in qpos
        qvel_offset: int = 6, # index where joint velocities start in qvel
    ) -> np.ndarray:
        """
        Compute PD torques from normalized action.

        Returns:
            torques: shape [nu], clipped to torque_limit
        """
        # Target joint positions: q_default + action * action_scale
        q_target = self._q_default + action * self.action_scale

        # Clip to joint range
        q_target = np.clip(q_target, self._jnt_range[:, 0], self._jnt_range[:, 1])

        # Current joint state
        q_curr  = q[qpos_offset: qpos_offset + self.nu]
        dq_curr = dq[qvel_offset: qvel_offset + self.nu]

        # PD law (joint-space torques)
        torques = self.kp * (q_target - q_curr) - self.kd * dq_curr

        # Clip to torque limits (joint space)
        torques = np.clip(torques, -self.torque_limit, self.torque_limit)
        # Convert to motor ctrl space: ctrl = torque / gear
        # MuJoCo motor actuator: force_applied = ctrl * gear
        torques = torques / self._gear
        return torques.astype(np.float32)
