"""
humanoid_rl/envs/mujoco/h1_env.py
Unitree H1 Humanoid Environment — production-quality locomotion env.

Key improvements over base env:
  - PD position control (action = target joint positions)
  - Terrain injection (stair, rough, slope, hurdle, gap)
  - Domain randomization (mass, friction, damping, gear, pushes)
  - Rich observation: qpos + qvel + body heights + contacts + prev_action
  - Per-task reward weights (tuned for H1)
  - Proprioceptive history (optional)
"""
from __future__ import annotations

import mujoco
import numpy as np
from pathlib import Path
from gymnasium import spaces
from typing import Any

from humanoid_rl.envs.base_env import BaseHumanoidEnv
from humanoid_rl.envs.mujoco.pd_control import PDController
from humanoid_rl.envs.mujoco.terrain import inject_terrain
from humanoid_rl.envs.mujoco.domain_rand import DomainRandomizer, PushRandomizer
from humanoid_rl.rewards.locomotion import LocomotionRewards
from humanoid_rl.rewards.stability import StabilityRewards
from humanoid_rl.rewards.energy import EnergyRewards
from humanoid_rl.rewards.gait import GaitRewards


# ─── H1 Physical Constants ────────────────────────────────────────────────────

# Per-joint PD gains for Unitree H1
# Tuned for position control: kp creates restoring torque, kd damps oscillation.
# H1 Menagerie model has joint damping=1.0 and armature=0.1.
# These PD gains are applied ON TOP of model dynamics via data.ctrl (direct torque).
H1_KP = np.array([
    # hip_yaw, hip_roll, hip_pitch, knee, ankle (×2 sides)
    200.0, 200.0, 200.0, 300.0, 40.0,   # left leg
    200.0, 200.0, 200.0, 300.0, 40.0,   # right leg
    200.0,                               # torso
    40.0,  40.0,  18.0,  18.0,          # left arm
    40.0,  40.0,  18.0,  18.0,          # right arm
], dtype=np.float32)

H1_KD = np.array([
    5.0,  5.0,  8.0, 10.0,  2.0,        # left leg
    5.0,  5.0,  8.0, 10.0,  2.0,        # right leg
    5.0,                                 # torso
    2.0,  2.0,  1.0,  1.0,              # left arm
    2.0,  2.0,  1.0,  1.0,              # right arm
], dtype=np.float32)

# Torque limits (Nm) — matched to H1 Menagerie actuator ctrlrange
# MuJoCo clamps data.ctrl to actuator_ctrlrange, so these must not exceed it.
H1_TORQUE_LIMITS = np.array([
    200.0, 200.0, 200.0, 300.0, 40.0,   # left leg
    200.0, 200.0, 200.0, 300.0, 40.0,   # right leg
    200.0,                               # torso
    40.0,  40.0,  18.0,  18.0,          # left arm  (ctrlrange=±40,±40,±18,±18)
    40.0,  40.0,  18.0,  18.0,          # right arm
], dtype=np.float32)

# Default standing pose (radians)
# Leg joints keep hip_pitch + knee + ankle = 0 (flat foot). The old
# (-0.2, 0.2, 0.0) put the feet ~8 cm in front of the hip, i.e. the CoM
# started ~10 cm behind the heel and the robot always fell backwards.
# (-0.15, 0.4, -0.25) was chosen by a sweep (CoM ~5 cm behind ankle-centre,
# longest passive survival) -- see tests/test_training_correctness.py.
H1_DEFAULT_QPOS = np.array([
    0.0, 0.0, -0.15, 0.4, -0.25,  # left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
    0.0, 0.0, -0.15, 0.4, -0.25,  # right leg
    0.0,                                 # torso
    0.0, 0.2, 0.0, 0.3,            # left arm
    0.0, -0.2, 0.0, 0.3,           # right arm
], dtype=np.float32)

# Root height: Adjusted for crouched pose
H1_INIT_ROOT_Z = 1.025  # m (feet just touch the floor with H1_DEFAULT_QPOS)

# No forward lean needed — with straight legs, CoM is naturally over feet.
# Quaternion: identity = [1, 0, 0, 0]
H1_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Action scale: max joint deviation from default (radians)
# With kp=200 Nm/rad, 0.3 rad → 60 Nm max PD torque (within ctrlrange for legs)
H1_ACTION_SCALE = 0.3  # rad

# Per-joint action scale. Ankles need far more authority: with 0.3 rad * kp 40
# they can only produce 12 Nm, but holding the CoM ~5 cm off the ankle already
# needs ~12 Nm per ankle plus margin for corrections. A random linear balance
# controller survives 500/500 steps with ankle scale 1.0 (=40 Nm torque limit)
# and at most 162/500 with 0.3.
H1_ACTION_SCALE_VEC = np.full(19, H1_ACTION_SCALE, dtype=np.float32)
H1_ACTION_SCALE_VEC[[4, 9]] = 1.0   # left/right ankle

# Height of the ankle link origin above ground with the foot flat (measured).
# Foot clearance = ankle_z - this.
H1_FOOT_REST_Z = 0.068  # m (settled on flat ground)

# Middle of the foot sole in front of the ankle origin (heel -0.035, toe +0.17).
H1_SUPPORT_OFFSET_X = 0.06  # m


class H1Env(BaseHumanoidEnv):
    """
    Unitree H1 Humanoid Locomotion Environment.

    Uses PD position control, terrain injection, and domain randomization.

    Args:
        task_cfg:         Task configuration dict (from task.to_cfg())
        terrain_type:     "flat" | "stair" | "rough" | "slope" | "hurdle" | "gap"
        terrain_kwargs:   Extra args for terrain generator
        domain_rand_cfg:  Domain randomization config dict (None = disabled)
        push_rand_cfg:    Push randomization config dict (None = disabled)
        render_mode:      "human" | "rgb_array" | None
        frame_skip:       Physics steps per control step (default 10 → 50Hz)
        width, height:    Render resolution
        history_len:      Proprioceptive history frames (1 = no history)
    """

    def __init__(
        self,
        task_cfg:        dict,
        terrain_type:    str | None = None,
        terrain_kwargs:  dict | None = None,
        domain_rand_cfg: dict | None = None,
        push_rand_cfg:   dict | None = None,
        render_mode:     str | None = None,
        frame_skip:      int = 10,
        width:           int = 640,
        height:          int = 480,
        history_len:     int = 1,
    ):
        from humanoid_rl.robots.unitree_h1 import get_h1_cfg
        robot_cfg = get_h1_cfg()

        super().__init__(robot_cfg, task_cfg, render_mode, frame_skip)

        self._width      = width
        self._height     = height
        self._history_len = history_len

        # ── Load MuJoCo model with terrain ──────────────────────────────────
        xml_path = robot_cfg["mjcf_path_absolute"]
        xml_dir  = str(Path(xml_path).parent)

        with open(xml_path, "r") as f:
            base_xml = f.read()

        # Fix asset paths to absolute so from_xml_string works
        base_xml = base_xml.replace(
            'meshdir="assets"',
            f'meshdir="{xml_dir}/assets"'
        ).replace(
            'texturedir="assets"',
            f'texturedir="{xml_dir}/assets"'
        )

        # Patch offscreen framebuffer — MUST happen before MjModel.from_xml_string
        import re as _re
        fb_size = max(width, height, 640)
        visual_block = f'  <visual>\n    <global offwidth="{fb_size}" offheight="{fb_size}"/>\n  </visual>'
        if _re.search(r'<visual\b', base_xml):
            if 'offwidth' in base_xml:
                base_xml = _re.sub(r'offwidth="\d+"', f'offwidth="{fb_size}"', base_xml)
                base_xml = _re.sub(r'offheight="\d+"', f'offheight="{fb_size}"', base_xml)
            else:
                base_xml = _re.sub(
                    r'(<visual[^>]*>)',
                    r'\1\n    ' + f'<global offwidth="{fb_size}" offheight="{fb_size}"/>',
                    base_xml
                )
        else:
            base_xml = _re.sub(
                r'(<mujoco[^>]*>)',
                r'\1\n' + visual_block,
                base_xml, count=1
            )
        if terrain_type is None:
            terrain_type = task_cfg.get("terrain", "flat")
            if terrain_kwargs is None:
                terrain_kwargs = task_cfg.get("terrain_kwargs")
        self._terrain_type = terrain_type
        terrain_xml = inject_terrain(
            base_xml,
            terrain_type,
            **(terrain_kwargs or {})
        )

        self._model = mujoco.MjModel.from_xml_string(terrain_xml)
        self._data  = mujoco.MjData(self._model)

        self._sim_dt  = self._model.opt.timestep
        self._ctrl_dt = self._sim_dt * frame_skip

        nu = self._model.nu
        assert nu == 19, f"H1 should have 19 actuators, got {nu}"

        # ── PD Controller ───────────────────────────────────────────────────
        self._pd = PDController(
            model        = self._model,
            kp           = H1_KP,
            kd           = H1_KD,
            torque_limit = H1_TORQUE_LIMITS,
            action_scale = H1_ACTION_SCALE_VEC,
        )
        self._pd.set_default_pose(
            np.concatenate([np.zeros(7), H1_DEFAULT_QPOS])
        )

        # ── Domain Randomization ─────────────────────────────────────────────
        self._domain_rand = None
        self._push_rand   = None
        if domain_rand_cfg is not None:
            self._domain_rand = DomainRandomizer(self._model, domain_rand_cfg)
        if push_rand_cfg is not None:
            self._push_rand = PushRandomizer(push_rand_cfg)
        self._rng = np.random.default_rng()

        # ── Observation Space ────────────────────────────────────────────────
        obs_dim = self._compute_obs_dim()
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(nu,), dtype=np.float32
        )

        # ── Body names (H1-specific) ─────────────────────────────────────────
        self._torso_body  = "torso_link"
        self._pelvis_body = "pelvis"
        self._lfoot_body  = "left_ankle_link"
        self._rfoot_body  = "right_ankle_link"

        # Cache body IDs for balance force + contacts
        self._robot_root_bid = 1  # body 1 = robot root (pelvis)
        self._pelvis_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._pelvis_body)
        self._lfoot_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._lfoot_body)
        self._rfoot_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._rfoot_body)

        # ── Reward modules ───────────────────────────────────────────────────
        reward_cfg = task_cfg.get("rewards", {})
        self._loco_rewards   = LocomotionRewards(reward_cfg.get("locomotion", {}))
        self._stab_rewards   = StabilityRewards(reward_cfg.get("stability", {}))
        self._energy_rewards = EnergyRewards(reward_cfg.get("energy", {}))
        # Gait rewards: only active if "gait" key present in task config
        gait_cfg = reward_cfg.get("gait", None)
        self._gait_rewards = (
            GaitRewards(gait_cfg, control_dt=self._ctrl_dt, default_q=H1_DEFAULT_QPOS)
            if gait_cfg else None
        )
        self._reset_pending = False  # True only for the obs returned by reset()

        # ── Task params ──────────────────────────────────────────────────────
        self._max_steps    = task_cfg.get("max_steps", 1000)
        self._target_speed = task_cfg.get("target_speed", 1.0)

        # ── State ────────────────────────────────────────────────────────────
        self._renderer  = None
        self._viewer    = None
        self._prev_action = np.zeros(nu, dtype=np.float32)
        self._obs_history = None  # for history_len > 1

        mujoco.mj_forward(self._model, self._data)
        self._init_qpos = self._data.qpos.copy()
        self._init_qvel = self._data.qvel.copy()

    # ── Observation ──────────────────────────────────────────────────────────

    def _compute_obs_dim(self) -> int:
        """
        H1 Observation (per frame):
          qpos[2:]           = 24  (z + quat[4] + joints[19])
          qvel               = 25
          body_heights       = 4   (torso, pelvis, left_ankle, right_ankle z)
          foot_contacts      = 2
          com_rel            = 2   (CoM x,y relative to support center)
          com_vel            = 2   (CoM velocity x,y)
          root_orient        = 2   (roll, pitch)
          root_ang_vel       = 2   (roll_rate, pitch_rate)
          prev_action        = 19
          ─────────────────────────
          per-frame total    = 82
          × history_len      = 82 * history_len
        """
        nu = self._model.nu  # 19
        per_frame = (
            (self._model.nq - 2) +  # qpos: skip root x,y → 24
            self._model.nv +         # qvel: 25
            4 +                      # body heights
            2 +                      # foot contacts
            2 +                      # com_rel (CoM relative to support center)
            2 +                      # com_vel (CoM velocity)
            2 +                      # root_orient (roll, pitch)
            2 +                      # root_ang_vel (roll_rate, pitch_rate)
            nu                       # prev_action
        )
        if self.task_cfg.get("rewards", {}).get("gait"):
            per_frame += 2           # gait phase (sin, cos)
        return per_frame * self._history_len

    def _get_obs_frame(self) -> np.ndarray:
        """Compute single observation frame."""
        d = self._data

        # Joint positions: skip x, y (indices 0,1) — keep z, quat, joints
        qpos = d.qpos[2:].copy()  # [24]

        # Joint velocities (all)
        qvel = d.qvel.copy()      # [25]

        # Body heights
        torso_z  = float(self._get_body_z(self._torso_body))
        pelvis_z = float(self._get_body_z(self._pelvis_body))
        lfoot_z  = float(self._get_body_z(self._lfoot_body))
        rfoot_z  = float(self._get_body_z(self._rfoot_body))
        body_heights = np.array([torso_z, pelvis_z, lfoot_z, rfoot_z], dtype=np.float32)

        # Foot contacts
        contacts = self._get_foot_contacts()

        # ── Balance observations ─────────────────────────────────────────
        # CoM position relative to support center
        com_rel, com_vel_xy = self._compute_com_state()
        # Root orientation: roll & pitch from quaternion
        qw, qx, qy, qz = d.qpos[3:7]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))
        root_orient = np.array([roll, pitch], dtype=np.float32)
        # Root angular velocity
        root_ang_vel = np.array([d.qvel[3], d.qvel[4]], dtype=np.float32)  # roll_rate, pitch_rate

        # Previous action (proprioceptive)
        prev_a = self._prev_action.copy()

        parts = [
            qpos, qvel, body_heights, contacts,
            com_rel, com_vel_xy, root_orient, root_ang_vel,
            prev_a,
        ]
        if self._gait_rewards is not None:
            # Reward of the NEXT step uses step_count = self._step_count + 1
            # (it is incremented after obs/reward); reset obs is phase of step 0.
            nxt = self._step_count if self._reset_pending else self._step_count + 1
            parts.append(self._gait_rewards.phase_features(nxt))

        return np.concatenate(parts, dtype=np.float32)

    def _compute_com_state(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute center-of-mass position relative to support center
        and CoM horizontal velocity.

        Returns:
            com_rel:    [com_x - sup_x, com_y - sup_y]  (2,)
            com_vel_xy: [com_vx, com_vy]                 (2,)
        """
        m = self._model
        d = self._data
        com = np.zeros(3)
        com_vel = np.zeros(3)
        total_m = 0.0
        for i in range(1, m.nbody):
            mi = m.body_mass[i]
            # CoM position — xipos is body CoM in world frame (CORRECT)
            com += d.xipos[i] * mi
            # CoM velocity — mj_objectVelocity returns body frame origin,
            # but CoM can be offset. Fix: v_com = v_frame + omega * r_offset
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, i, v6, 0)
            r_offset = d.xipos[i] - d.xpos[i]
            body_com_vel = v6[3:6] + np.cross(v6[0:3], r_offset)
            com_vel += body_com_vel * mi
            total_m += mi
        com /= total_m
        com_vel /= total_m

        # Support center: midpoint of ankles, shifted to the middle of the foot
        # (foot extends ~-3.5 cm heel .. +17 cm toe from the ankle origin),
        # along the pelvis heading.
        lank_pos = self._get_body_pos(self._lfoot_body)
        rank_pos = self._get_body_pos(self._rfoot_body)
        sup = (lank_pos + rank_pos) / 2.0
        qw, qx, qy, qz = d.qpos[3:7]
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        sup[0] += H1_SUPPORT_OFFSET_X * np.cos(yaw)
        sup[1] += H1_SUPPORT_OFFSET_X * np.sin(yaw)

        com_rel = np.array([com[0] - sup[0], com[1] - sup[1]], dtype=np.float32)
        com_vel_xy = np.array([com_vel[0], com_vel[1]], dtype=np.float32)
        return com_rel, com_vel_xy

    def _get_obs(self) -> np.ndarray:
        """Get full observation (with history if enabled)."""
        frame = self._get_obs_frame()
        if self._history_len == 1:
            return frame
        # Shift history and add new frame
        if self._obs_history is None:
            self._obs_history = np.tile(frame, (self._history_len, 1))
        else:
            self._obs_history = np.roll(self._obs_history, -1, axis=0)
            self._obs_history[-1] = frame
        return self._obs_history.flatten()

    def _get_body_z(self, name: str) -> float:
        try:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            return float(self._data.xpos[bid, 2])
        except Exception:
            return 0.0

    def _body_exists(self, name: str) -> bool:
        try:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            return bid >= 0
        except Exception:
            return False

    def _get_body_pos(self, name: str) -> np.ndarray:
        try:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            return self._data.xpos[bid].copy()
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _get_foot_contacts(self) -> np.ndarray:
        """
        Return binary foot contact indicators using body-based approach.

        Uses rootid comparison to detect floor contacts: any body whose
        root body is NOT the robot root is considered "ground". This is
        more robust than checking body==0 (world body) because some models
        may have additional kinematic chains attached to world.
        """
        contacts = np.zeros(2, dtype=np.float32)

        # Pre-resolve body IDs (cached)
        if not hasattr(self, '_lfoot_bid'):
            self._robot_root_bid = 1  # body 1 = robot root (pelvis)
            try:
                self._lfoot_bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, self._lfoot_body)
            except Exception:
                self._lfoot_bid = -1
            try:
                self._rfoot_bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, self._rfoot_body)
            except Exception:
                self._rfoot_bid = -1

        for i in range(self._data.ncon):
            c = self._data.contact[i]
            # Resolve geom → body
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            # Ground contact: the geom that is NOT part of the robot subtree
            b1_is_robot = self._model.body_rootid[b1] == self._robot_root_bid
            b2_is_robot = self._model.body_rootid[b2] == self._robot_root_bid
            touching_ground = b1_is_robot != b2_is_robot  # one robot, one ground
            if not touching_ground:
                continue

            other_body = b1 if b1_is_robot else b2

            if other_body == self._lfoot_bid:
                contacts[0] = 1.0
            elif other_body == self._rfoot_bid:
                contacts[1] = 1.0

        return contacts

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def observation_space(self): return self._observation_space

    @property
    def action_space(self): return self._action_space

    @property
    def dt(self): return self._ctrl_dt

    @property
    def model(self): return self._model

    @property
    def data(self): return self._data


    # ── Virtual balance force ───────────────────────────────────────────────
    # PD on CoM position + orientation, applied to pelvis.
    # Same mechanism as play_h1.py -- keeps the inverted pendulum upright.
    # Without this, PD-only joints cannot fight gravity on the pendulum.

    def _step_simulation(self, action: np.ndarray) -> None:
        """PD control: action = normalized joint offsets [-1,1] -> torques + balance force."""
        for _ in range(self.frame_skip):
            # PD torques
            torques = self._pd.compute_torques(
                action     = action,
                q          = self._data.qpos,
                dq         = self._data.qvel,
                qpos_offset = 7,
                qvel_offset = 6,
            )
            self._data.ctrl[:] = torques

            # Balance force (computed every frame like play_h1.py)
            self._apply_balance_force()

            if self._push_rand is not None:
                self._push_rand.step(self._model, self._data, self._rng)

            mujoco.mj_step(self._model, self._data)

        self._prev_action[:] = action

    def _apply_balance_force(self) -> None:
        """
        Balance force disabled.

        The MuJoCo physics for H1 humanoid is inherently unstable with frame_skip > 1.
        For RL training, the policy needs to learn balance through joint control.
        The termination conditions will end episodes when the robot falls.
        """
        # Zero out any applied forces
        self._data.xfrc_applied[self._pelvis_bid, :] = [0, 0, 0, 0, 0, 0]

    # ── Reward ───────────────────────────────────────────────────────────────

    def _get_reward(self, action: np.ndarray) -> float:
        state = self._build_state_dict(action)
        r  = self._loco_rewards.compute(state)
        r += self._stab_rewards.compute(state)
        r += self._energy_rewards.compute(state)
        if self._gait_rewards is not None:
            r += self._gait_rewards.compute(state)
        return float(r)

    def _build_state_dict(self, action: np.ndarray) -> dict:
        d = self._data
        try:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, self._torso_body)
            upright_vec = d.xmat[bid].reshape(3, 3)[:, 2]
        except Exception:
            upright_vec = np.array([0.0, 0.0, 1.0])

        contacts = self._get_foot_contacts()

        # Foot velocities (for gait reward: foot slip penalty)
        def _foot_vel(body_name: str) -> np.ndarray:
            try:
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(
                    self._model, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0
                )
                return vel[3:6]   # linear velocity (x,y,z)
            except Exception:
                return np.zeros(3)

        # Balance state
        com_rel, com_vel_xy = self._compute_com_state()
        qw, qx, qy, qz = d.qpos[3:7]
        roll  = float(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2)))
        pitch = float(np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2)))

        return {
            "root_vel":       d.qvel[:3].copy(),
            "root_ang_vel":   d.qvel[3:6].copy(),
            "torso_height":   float(self._get_body_z(self._torso_body)),
            "upright_vec":    upright_vec,
            "torques":        d.actuator_force.copy(),
            "action":         action,
            "target_speed":   self._target_speed,
            "foot_contacts":  contacts,
            "alive":          not self._is_terminated(),
            # Balance observations
            "com_rel":        com_rel,          # [2] CoM relative to support center
            "com_vel_xy":     com_vel_xy,       # [2] CoM horizontal velocity
            "roll":           roll,
            "pitch":          pitch,
            "roll_rate":      float(d.qvel[3]),
            "pitch_rate":     float(d.qvel[4]),
            # Gait reward extras
            "q_joints":       d.qpos[7:].copy(),       # [19] joint angles
            "lfoot_z":        max(float(self._get_body_z(self._lfoot_body)) - H1_FOOT_REST_Z, 0.0),
            "rfoot_z":        max(float(self._get_body_z(self._rfoot_body)) - H1_FOOT_REST_Z, 0.0),
            "lfoot_contact":  bool(contacts[0] > 0.5),
            "rfoot_contact":  bool(contacts[1] > 0.5),
            "lfoot_vel":      _foot_vel(self._lfoot_body),
            "rfoot_vel":      _foot_vel(self._rfoot_body),
            "step_count":     self._step_count,
            "vx":             float(d.qvel[0]),
        }

    # ── Termination ──────────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        torso_z = self._get_body_z(self._torso_body)
        min_h = self.robot_cfg.get("min_torso_height", 0.6)
        if np.isnan(torso_z) or torso_z < min_h:
            return True
        if np.any(np.isnan(self._data.qpos)):
            return True
        # Orientation termination
        qw, qx, qy, qz = self._data.qpos[3:7]
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        if abs(pitch) > 0.8 or abs(roll) > 0.8:
            return True
        # Joint limit violation
        for jid in range(self._model.njnt):
            if not self._model.jnt_limited[jid]:
                continue
            qpos_adr = self._model.jnt_qposadr[jid]
            if qpos_adr < 7:
                continue
            lo, hi = self._model.jnt_range[jid]
            if lo >= hi:
                continue
            pos = self._data.qpos[qpos_adr]
            margin = 0.1 * (hi - lo)
            if pos < lo - margin or pos > hi + margin:
                return True
        return False

    def _is_truncated(self) -> bool:
        return self._step_count >= self._max_steps

    def _get_info(self) -> dict[str, Any]:
        return {
            "torso_height":  float(self._get_body_z(self._torso_body)),
            "x_velocity":    float(self._data.qvel[0]),
            "y_velocity":    float(self._data.qvel[1]),
            "foot_contacts": self._get_foot_contacts().tolist(),
        }

    # ── Reset ─────────────────────────────────────────────────────────────────

    def _reset_model(self) -> np.ndarray:
        noise = self.robot_cfg.get("reset_noise_scale", 0.01)
        rng   = self.np_random

        # Reset to slightly perturbed default standing pose
        self._data.qpos[:] = self._init_qpos.copy()

        # Set root height so feet contact floor
        self._data.qpos[2] = H1_INIT_ROOT_Z

        # Set root orientation (identity + small noise)
        self._data.qpos[3:7] = H1_INIT_QUAT + rng.uniform(-0.005, 0.005, 4)
        q = self._data.qpos[3:7]
        self._data.qpos[3:7] = q / np.linalg.norm(q)

        # Set joint positions to default + noise
        self._data.qpos[7:] = H1_DEFAULT_QPOS + rng.uniform(-noise, noise, 19)
        self._data.qvel[:] = self._init_qvel + rng.uniform(-noise * 0.1, noise * 0.1, self._model.nv)

        # Domain randomization
        if self._domain_rand is not None:
            self._domain_rand.randomize(self._model, self._rng)

        # Reset push state
        if self._push_rand is not None:
            self._push_rand.reset(self._data)

        # Reset history
        self._obs_history = None
        self._prev_action[:] = 0.0

        # Reset reward state (all use .reset(), no replacement)
        self._loco_rewards.reset()
        self._stab_rewards.reset()
        self._energy_rewards.reset()
        if self._gait_rewards is not None:
            self._gait_rewards.reset()

        mujoco.mj_forward(self._model, self._data)
        self._reset_pending = True
        try:
            return self._get_obs()
        finally:
            self._reset_pending = False



    # ── Render ────────────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                try:
                    from mujoco import viewer
                    self._viewer = viewer.launch_passive(self._model, self._data)
                except Exception as e:
                    print(f"[Render] Cannot open viewer (no display?): {e}")
                    self._viewer = None
                    self.render_mode = None  # disable further attempts
                    return
            try:
                self._viewer.sync()
            except Exception:
                self._viewer = None
                self.render_mode = None
        elif self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self._model, self._height, self._width)
            self._renderer.update_scene(self._data)
            return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        self._renderer = None
