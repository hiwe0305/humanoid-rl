"""
humanoid_rl/envs/mujoco/balance_controller.py
Active balance controller for standing humanoid.

Implements ankle strategy + hip strategy to maintain center of mass (CoM)
over the base of support. This works as a feedforward term on top of PD control.

Physics:
    For an inverted pendulum of mass m, height h, with CoM offset Δx from ankle:
    Required ankle torque = m × g × Δx (per ankle, divide by 2 for biped)
    
    Additional damping on root pitch velocity prevents oscillation.
"""
from __future__ import annotations

import numpy as np
import mujoco


class BalanceController:
    """
    CoM-based balance controller for bipedal standing.
    
    Computes additional torques for ankle and hip joints to keep the
    center of mass (CoM) projected within the base of support.
    
    Args:
        model:   MuJoCo model
        kp_com:  proportional gain on CoM error (Nm/m)
        kd_com:  derivative gain on CoM velocity (Nm·s/m)
        kp_pitch: proportional gain on root pitch (Nm/rad)
        kd_pitch: derivative gain on root pitch rate (Nm·s/rad)
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        kp_com:   float = 300.0,   # strong CoM position feedback
        kd_com:   float = 50.0,    # damp CoM velocity
        kp_pitch: float = 100.0,   # correct body lean
        kd_pitch: float = 20.0,    # damp pitch oscillation
    ):
        self.model    = model
        self.kp_com   = kp_com
        self.kd_com   = kd_com
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        
        # Cache body IDs
        self._torso_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self._pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._lankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
        self._rankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")
        
        # Total mass (for gravity compensation)
        self._total_mass = sum(model.body_mass[i] for i in range(1, model.nbody))
        self._gravity = abs(model.opt.gravity[2])  # 9.81
        
    def compute_balance_torques(
        self,
        data: mujoco.MjData,
        nu: int = 19,
    ) -> np.ndarray:
        """
        Compute balance correction torques.
        
        Returns:
            torques: shape [nu], to be ADDED to PD torques
        """
        balance = np.zeros(nu, dtype=np.float32)
        
        # ── 1. Compute CoM position ──────────────────────────────────
        com = np.zeros(3)
        com_vel = np.zeros(3)
        total_mass = 0.0
        for i in range(1, self.model.nbody):
            m = self.model.body_mass[i]
            com += data.xipos[i] * m
            # Body velocity
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, i, vel, 0
            )
            com_vel += vel[3:6] * m  # linear velocity
            total_mass += m
        com /= total_mass
        com_vel /= total_mass
        
        # ── 2. Compute support center (midpoint of ankles) ──────────
        ankle_l = data.xpos[self._lankle_id]
        ankle_r = data.xpos[self._rankle_id]
        support_center = (ankle_l + ankle_r) / 2.0
        
        # CoM error relative to support center (in world x direction = forward)
        com_err_x = com[0] - support_center[0]   # positive = CoM forward of ankles
        com_err_y = com[1] - support_center[1]   # positive = CoM left
        com_vel_x = com_vel[0]
        com_vel_y = com_vel[1]
        
        # ── 3. Root pitch angle and rate ─────────────────────────────
        # From quaternion: extract pitch (rotation about y-axis)
        qw, qx, qy, qz = data.qpos[3:7]
        # Pitch = atan2(2(qw*qy - qz*qx), 1 - 2(qx² + qy²))
        pitch = np.arctan2(2 * (qw * qy - qz * qx), 
                           1 - 2 * (qx**2 + qy**2))
        pitch_rate = data.qvel[4]  # angular velocity around y-axis
        
        # ── 4. Ankle strategy (sagittal balance) ─────────────────────
        # Ankle torque to bring CoM over support center
        # τ_ankle = -kp * com_err_x - kd * com_vel_x
        # Negative sign: if CoM is forward, push ankle backward (plantarflex)
        ankle_torque = -(self.kp_com * com_err_x + self.kd_com * com_vel_x)
        
        # Split equally between left and right ankle
        # Joint indices: left_ankle=4, right_ankle=9
        ankle_torque_each = np.clip(ankle_torque / 2.0, -35.0, 35.0)
        balance[4] = ankle_torque_each
        balance[9] = ankle_torque_each
        
        # ── 5. Hip strategy (pitch correction) ───────────────────────
        # Hip pitch torque to correct body lean
        # If body pitches forward (positive pitch), flex hips backward
        hip_torque = -(self.kp_pitch * pitch + self.kd_pitch * pitch_rate)
        hip_torque_each = np.clip(hip_torque / 2.0, -80.0, 80.0)
        
        # Joint indices: left_hip_pitch=2, right_hip_pitch=7
        balance[2] = hip_torque_each
        balance[7] = hip_torque_each
        
        # ── 6. Lateral balance (roll) ────────────────────────────────
        # Roll angle
        roll = np.arctan2(2 * (qw * qx + qy * qz),
                          1 - 2 * (qx**2 + qy**2))  # fixed: this is actually roll
        roll_rate = data.qvel[3]
        
        # Hip roll correction
        roll_torque = -(60.0 * roll + 10.0 * roll_rate)
        roll_torque_each = np.clip(roll_torque / 2.0, -40.0, 40.0)
        
        # Joint indices: left_hip_roll=1, right_hip_roll=6
        balance[1] = roll_torque_each
        balance[6] = roll_torque_each
        
        # ── 7. Knee anti-buckle ──────────────────────────────────────
        # Prevent knees from collapsing under weight
        # Add small extension torque proportional to gravity
        knee_support = self._total_mass * self._gravity * 0.01  # ~5 Nm
        balance[3] = -knee_support   # left knee: resist flexion
        balance[8] = -knee_support   # right knee
        
        return balance
