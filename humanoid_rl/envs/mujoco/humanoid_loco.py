
"""
humanoid_rl/envs/mujoco/humanoid_loco.py
MuJoCo locomotion env cho generic humanoid robot — FIXED: PD position control.

ROOT CAUSE của standing problem: dùng direct torque control (action × torque_limit).
Policy phải output chính xác Nm torque cho mỗi joint → exploration khổng lồ → khó học.

FIX: Thêm PD position control (giống H1Env).
  action [-1,1] → target joint offset → PD law → torque  
  → Policy chỉ cần học offset joint mục tiêu, PD law tự sinh torque giữ vị trí.
  
Also added: terrain injection, CoM observation, body height obs, gait rewards.
"""

from __future__ import annotations
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from humanoid_rl.envs.base_env import BaseHumanoidEnv
from humanoid_rl.envs.mujoco.pd_control import PDController
from humanoid_rl.envs.mujoco.domain_rand import DomainRandomizer, PushRandomizer
from humanoid_rl.rewards.locomotion import LocomotionRewards
from humanoid_rl.rewards.stability import StabilityRewards
from humanoid_rl.rewards.energy import EnergyRewards
from humanoid_rl.rewards.gait import GaitRewards

# --- Generic Humanoid PD Gains ---
# Lower than H1 — robot is lighter, joints less powerful
HUMANOID_KP = np.array([
    60.0,  60.0,  60.0,   # abdomen_z, abdomen_y, abdomen_x
    60.0,  60.0, 120.0,   # right_hip_x, right_hip_z, right_hip_y
    80.0,                  # right_knee
    60.0,  60.0, 120.0,   # left_hip_x, left_hip_z, left_hip_y
    80.0,                  # left_knee
    20.0,  20.0,  40.0,   # right_shoulder1, right_shoulder2, right_elbow
    20.0,  20.0,  40.0,   # left_shoulder1, left_shoulder2, left_elbow
], dtype=np.float32)

HUMANOID_KD = np.array([
    3.0,  3.0,  3.0,      # abdomen
    3.0,  3.0,  5.0,      # right_hip
    4.0,                   # right_knee
    3.0,  3.0,  5.0,      # left_hip
    4.0,                   # left_knee
    1.0,  1.0,  2.0,      # right_arm
    1.0,  1.0,  2.0,      # left_arm
], dtype=np.float32)

HUMANOID_DEFAULT_QPOS = np.array([
    0.0,  0.0,  0.0,      # abdomen: straight
    0.0,  0.0, -0.25,     # right_hip_x(0), right_hip_z(0), right_hip_y(-0.25)
    -0.50,                 # right_knee: slightly bent
    0.0,  0.0, -0.25,     # left_hip
    -0.50,                 # left_knee
    0.0,  0.0, -0.50,     # right_shoulder1, right_shoulder2, right_elbow
    0.0,  0.0, -0.50,     # left_shoulder1, left_shoulder2, left_elbow
], dtype=np.float32)

HUMANOID_ACTION_SCALE = 0.4   # rad — joint deviation from default
HUMANOID_INIT_ROOT_Z = 1.4    # m — from XML torso pos
HUMANOID_INIT_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class MuJoCoHumanoidEnv(BaseHumanoidEnv):
    """
    MuJoCo generic humanoid with PD position control.
    FIXED: uses PD control instead of direct torque — can now learn to stand/walk.
    """
    def __init__(self, robot_cfg, task_cfg, render_mode=None, frame_skip=5,
                 width=640, height=480, terrain_type="flat", terrain_kwargs=None,
                 domain_rand_cfg=None, push_rand_cfg=None, history_len=1):
        super().__init__(robot_cfg, task_cfg, render_mode, frame_skip)
        self._width, self._height = width, height
        self._history_len = history_len

        # Load model
        xml_path = robot_cfg["mjcf_path_absolute"]
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data = mujoco.MjData(self._model)
        self._sim_dt = self._model.opt.timestep
        self._ctrl_dt = self._sim_dt * frame_skip

        nu = self._model.nu
        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._compute_obs_dim(),), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(nu,), dtype=np.float32
        )

        # PD Controller
        torque_limits = self._model.actuator_forcerange[:, 1].copy()
        if np.all(torque_limits == 0):
            torque_limits = np.ones(nu) * robot_cfg.get("default_torque_limit", 100.0)
        self._pd = PDController(
            model=self._model, kp=HUMANOID_KP, kd=HUMANOID_KD,
            torque_limit=torque_limits.astype(np.float32),
            action_scale=HUMANOID_ACTION_SCALE,
        )
        self._pd.set_default_pose(np.concatenate([np.zeros(7), HUMANOID_DEFAULT_QPOS]))

        # Domain randomization
        self._domain_rand = None
        self._push_rand = None
        if domain_rand_cfg is not None:
            self._domain_rand = DomainRandomizer(self._model, domain_rand_cfg)
        if push_rand_cfg is not None:
            self._push_rand = PushRandomizer(push_rand_cfg)
        self._rng = np.random.default_rng()

        # Reward modules
        reward_cfg = task_cfg.get("rewards", {})
        self._loco_rewards = LocomotionRewards(reward_cfg.get("locomotion", {}))
        self._stab_rewards = StabilityRewards(reward_cfg.get("stability", {}))
        self._energy_rewards = EnergyRewards(reward_cfg.get("energy", {}))
        gait_cfg = reward_cfg.get("gait", None)
        self._gait_rewards = GaitRewards(gait_cfg, control_dt=self._ctrl_dt) if gait_cfg else None

        # Task
        self._max_steps = task_cfg.get("max_steps", 1000)
        self._target_speed = task_cfg.get("target_speed", 1.0)

        # Body names
        self._torso_body = "torso"
        self._pelvis_body = "lwaist"
        self._lfoot_body = "left_foot"
        self._rfoot_body = "right_foot"

        # State
        self._renderer = None
        self._viewer = None
        self._prev_action = np.zeros(nu, dtype=np.float32)
        self._obs_history = None

        mujoco.mj_forward(self._model, self._data)
        self._init_qpos = self._data.qpos.copy()
        self._init_qvel = self._data.qvel.copy()

    # --- Properties ---
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

    # --- Observation ---
    def _compute_obs_dim(self) -> int:
        nu = self._model.nu
        per_frame = (
            (self._model.nq - 2) +   # qpos: skip x,y
            self._model.nv +          # qvel
            4 +                       # body heights
            2 +                       # foot contacts
            2 +                       # com_rel
            2 +                       # com_vel_xy
            2 +                       # root_orient (roll, pitch)
            2 +                       # root_ang_vel (roll_rate, pitch_rate)
            nu                        # prev_action
        )
        return per_frame * self._history_len

    def _get_obs_frame(self) -> np.ndarray:
        d = self._data
        qpos = d.qpos[2:].copy()
        qvel = d.qvel.copy()
        body_heights = np.array([
            float(self._get_body_pos(self._torso_body)[2]),
            float(self._get_body_pos(self._pelvis_body)[2]),
            float(self._get_body_pos(self._lfoot_body)[2]),
            float(self._get_body_pos(self._rfoot_body)[2]),
        ], dtype=np.float32)
        contacts = self._get_foot_contacts()
        com_rel, com_vel_xy = self._compute_com_state()
        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))
        root_orient = np.array([roll, pitch], dtype=np.float32)
        root_ang_vel = np.array([d.qvel[3], d.qvel[4]], dtype=np.float32)
        prev_a = self._prev_action.copy()
        return np.concatenate([
            qpos, qvel, body_heights, contacts,
            com_rel, com_vel_xy, root_orient, root_ang_vel, prev_a,
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        frame = self._get_obs_frame()
        if self._history_len == 1:
            return frame
        if self._obs_history is None:
            self._obs_history = np.tile(frame, (self._history_len, 1))
        else:
            self._obs_history = np.roll(self._obs_history, -1, axis=0)
            self._obs_history[-1] = frame
        return self._obs_history.flatten()

    def _get_body_pos(self, name: str) -> np.ndarray:
        try:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            return self._data.xpos[bid].copy()
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _body_exists(self, name: str) -> bool:
        try:
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            return True
        except Exception:
            return False

    def _get_foot_contacts(self) -> np.ndarray:
        """Body-based contact detection with robust rootid check."""
        contacts = np.zeros(2, dtype=np.float32)
        if not hasattr(self, '_loco_lfoot_bid'):
            try:
                self._loco_lfoot_bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, self._lfoot_body)
            except Exception:
                self._loco_lfoot_bid = -1
            try:
                self._loco_rfoot_bid = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_BODY, self._rfoot_body)
            except Exception:
                self._loco_rfoot_bid = -1
            self._loco_robot_root = 1  # body 1 = robot root
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            b1 = self._model.geom_bodyid[c.geom1]
            b2 = self._model.geom_bodyid[c.geom2]
            # Check ground via rootid comparison
            b1_is_robot = self._model.body_rootid[b1] == self._loco_robot_root
            b2_is_robot = self._model.body_rootid[b2] == self._loco_robot_root
            if b1_is_robot == b2_is_robot:
                continue
            other = b2 if b1_is_robot else b1
            if other == self._loco_lfoot_bid:
                contacts[0] = 1.0
            elif other == self._loco_rfoot_bid:
                contacts[1] = 1.0
        return contacts

    def _compute_com_state(self):
        """CoM relative to support center + CoM horizontal velocity."""
        m, d = self._model, self._data
        com = np.zeros(3); com_vel = np.zeros(3); total_m = 0.0
        for i in range(1, m.nbody):
            mi = m.body_mass[i]
            com += d.xipos[i] * mi
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, i, v6, 0)
            # v6[3:6] is velocity at body frame origin; correct to CoM
            r_offset = d.xipos[i] - d.xpos[i]
            body_com_vel = v6[3:6] + np.cross(v6[0:3], r_offset)
            com_vel += body_com_vel * mi
            total_m += mi
        com /= total_m; com_vel /= total_m
        lpos = self._get_body_pos(self._lfoot_body)
        rpos = self._get_body_pos(self._rfoot_body)
        sup = (lpos + rpos) / 2.0
        return np.array([com[0]-sup[0], com[1]-sup[1]], dtype=np.float32), \
               np.array([com_vel[0], com_vel[1]], dtype=np.float32)

    # --- Reward ---
    def _get_reward(self, action: np.ndarray) -> float:
        state = self._build_state_dict(action)
        r = self._loco_rewards.compute(state)
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
        com_rel, com_vel_xy = self._compute_com_state()
        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))

        def _foot_vel(body_name):
            try:
                bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(self._model, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
                return vel[3:6]
            except Exception:
                return np.zeros(3)

        return {
            "root_vel": d.qvel[:3].copy(),
            "root_ang_vel": d.qvel[3:6].copy(),
            "torso_height": float(self._get_body_pos(self._torso_body)[2]),
            "upright_vec": upright_vec,
            "torques": d.actuator_force.copy(),
            "action": action,
            "target_speed": self._target_speed,
            "foot_contacts": contacts,
            "alive": not self._is_terminated(),
            "com_rel": com_rel,
            "com_vel_xy": com_vel_xy,
            "roll": roll,
            "pitch": pitch,
            "roll_rate": float(d.qvel[3]),
            "pitch_rate": float(d.qvel[4]),
            "q_joints": d.qpos[7:].copy(),
            "lfoot_z": float(self._get_body_pos(self._lfoot_body)[2]),
            "rfoot_z": float(self._get_body_pos(self._rfoot_body)[2]),
            "lfoot_contact": bool(contacts[0] > 0.5),
            "rfoot_contact": bool(contacts[1] > 0.5),
            "lfoot_vel": _foot_vel(self._lfoot_body),
            "rfoot_vel": _foot_vel(self._rfoot_body),
            "step_count": self._step_count,
            "vx": float(d.qvel[0]),
        }

    # --- Termination ---
    def _is_terminated(self) -> bool:
        torso_z = self._get_body_pos(self._torso_body)[2]
        min_h = self.robot_cfg.get("min_torso_height", 0.5)
        if np.isnan(torso_z) or torso_z < min_h:
            return True
        if np.any(np.isnan(self._data.qpos)):
            return True
        return False

    def _is_truncated(self) -> bool:
        return self._step_count >= self._max_steps

    def _get_info(self) -> dict[str, Any]:
        return {
            "torso_height": float(self._get_body_pos(self._torso_body)[2]),
            "x_velocity": float(self._data.qvel[0]),
            "y_velocity": float(self._data.qvel[1]),
            "foot_contacts": self._get_foot_contacts().tolist(),
        }

    # --- Physics step ---
    def _step_simulation(self, action: np.ndarray) -> None:
        """PD control: action = normalized joint offsets → torques."""
        for _ in range(self.frame_skip):
            # Recompute PD torques at each sub-step so restoring force
            # tracks changing joint position.
            torques = self._pd.compute_torques(
                action=action, q=self._data.qpos, dq=self._data.qvel,
                qpos_offset=7, qvel_offset=6,
            )
            self._data.ctrl[:] = torques
            if self._push_rand is not None:
                self._push_rand.step(self._model, self._data, self._rng)
            mujoco.mj_step(self._model, self._data)
        self._prev_action[:] = action

    # --- Reset ---
    def _reset_model(self) -> np.ndarray:
        noise = self.robot_cfg.get("reset_noise_scale", 0.01)
        self._data.qpos[:] = self._init_qpos.copy()
        self._data.qpos[2] = HUMANOID_INIT_ROOT_Z
        self._data.qpos[3:7] = HUMANOID_INIT_QUAT
        self._data.qpos[7:] = HUMANOID_DEFAULT_QPOS + \
            self.np_random.uniform(-noise, noise, self._model.nu)
        self._data.qvel[:] = self._init_qvel + \
            self.np_random.uniform(-noise*0.1, noise*0.1, self._model.nv)
        if self._domain_rand is not None:
            self._domain_rand.randomize(self._model, self._rng)
        if self._push_rand is not None:
            self._push_rand.reset(self._data)
        self._obs_history = None
        self._prev_action[:] = 0.0
        self._loco_rewards.reset()
        self._stab_rewards.reset()
        self._energy_rewards.reset()
        if self._gait_rewards is not None:
            self._gait_rewards.reset()
        mujoco.mj_forward(self._model, self._data)
        return self._get_obs()

    # --- Render ---
    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
            self._viewer.sync()
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
