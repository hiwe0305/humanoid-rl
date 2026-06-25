"""
humanoid_rl/envs/mujoco/humanoid_loco.py
Core MuJoCo locomotion environment for humanoid robots.

Wraps MuJoCo physics with:
  - Configurable robot (Unitree H1 or built-in humanoid)
  - Pluggable task (walk / run / stair / jump)
  - Modular reward composition
  - Real-time & offscreen rendering
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from humanoid_rl.envs.base_env import BaseHumanoidEnv
from humanoid_rl.rewards.locomotion import LocomotionRewards
from humanoid_rl.rewards.stability import StabilityRewards
from humanoid_rl.rewards.energy import EnergyRewards

# Path to MJCF assets directory
ASSETS_DIR = Path(__file__).parents[2] / "robots" / "assets"


class MuJoCoHumanoidEnv(BaseHumanoidEnv):
    """
    MuJoCo-based humanoid locomotion environment.

    Args:
        robot_cfg:    Robot configuration dict (from YAML)
        task_cfg:     Task configuration dict
        render_mode:  "human" | "rgb_array" | None
        frame_skip:   Physics steps per env step (default 5 → 100Hz/5=20Hz control)
        width, height: Render resolution
    """

    def __init__(
        self,
        robot_cfg: dict,
        task_cfg: dict,
        render_mode: str | None = None,
        frame_skip: int = 5,
        width: int = 640,
        height: int = 480,
    ):
        super().__init__(robot_cfg, task_cfg, render_mode, frame_skip)

        self._width  = width
        self._height = height

        # Load MuJoCo model
        mjcf_path = ASSETS_DIR / robot_cfg["mjcf_file"]
        if not mjcf_path.exists():
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")

        self._model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self._data  = mujoco.MjData(self._model)

        # Simulation dt
        self._sim_dt  = self._model.opt.timestep          # e.g. 0.002s
        self._ctrl_dt = self._sim_dt * frame_skip          # e.g. 0.01s (100Hz)

        # Spaces
        obs_dim = self._compute_obs_dim()
        act_dim = self._model.nu  # number of actuators

        self._observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
        )

        # Joint torque limits for action scaling
        self._torque_limits = self._model.actuator_forcerange[:, 1].copy()
        if np.all(self._torque_limits == 0):
            self._torque_limits = np.ones(act_dim) * robot_cfg.get("default_torque_limit", 100.0)

        # Reward modules
        reward_cfg = task_cfg.get("rewards", {})
        self._loco_rewards   = LocomotionRewards(reward_cfg.get("locomotion", {}))
        self._stab_rewards   = StabilityRewards(reward_cfg.get("stability", {}))
        self._energy_rewards = EnergyRewards(reward_cfg.get("energy", {}))

        # Task config
        self._max_steps    = task_cfg.get("max_steps", 1000)
        self._target_speed = task_cfg.get("target_speed", 1.0)   # m/s

        # Configurable body names (support both humanoid + H1)
        self._torso_body   = robot_cfg.get("torso_body_name",  "torso")
        self._pelvis_body  = robot_cfg.get("pelvis_body_name", "pelvis")
        self._lfoot_body   = robot_cfg.get("left_foot_body",   "left_foot")
        self._rfoot_body   = robot_cfg.get("right_foot_body",  "right_foot")

        # Renderer (lazy init)
        self._renderer = None
        self._viewer   = None

        # Initial qpos / qvel for reset
        self._init_qpos = self._data.qpos.copy()
        self._init_qvel = self._data.qvel.copy()

    # ─── Properties ────────────────────────────────────────────────────────

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def dt(self):
        return self._ctrl_dt

    @property
    def model(self):
        return self._model

    @property
    def data(self):
        return self._data

    # ─── Observation ───────────────────────────────────────────────────────

    def _compute_obs_dim(self) -> int:
        """
        Observation = qpos (excluding root x) + qvel + 4 body heights + 2 foot contacts
        body heights: torso_z, pelvis_z, left_foot_z, right_foot_z
        """
        nq = self._model.nq - 1   # exclude root x only
        nv = self._model.nv
        n_body_heights = 4        # torso, pelvis, left_foot, right_foot z
        n_contacts      = 2       # left/right foot contact booleans
        return nq + nv + n_body_heights + n_contacts

    def _get_obs(self) -> np.ndarray:
        d = self._data

        # qpos: skip root x (idx 0), keep root y-z + quaternion + joints
        qpos = d.qpos[1:].copy()   # remove x component

        # qvel: full
        qvel = d.qvel.copy()

        # Body heights (z-coordinates) — each must be a scalar float
        torso_body  = getattr(self, "_torso_body",  "torso")
        pelvis_body = getattr(self, "_pelvis_body", "pelvis")
        lfoot_body  = getattr(self, "_lfoot_body",  "left_foot")
        rfoot_body  = getattr(self, "_rfoot_body",  "right_foot")

        torso_z  = float(self._get_body_pos(torso_body)[2])
        pelvis_z = float(self._get_body_pos(pelvis_body)[2]) if self._body_exists(pelvis_body) else float(d.qpos[2])
        left_z   = float(self._get_body_pos(lfoot_body)[2])  if self._body_exists(lfoot_body)  else 0.0
        right_z  = float(self._get_body_pos(rfoot_body)[2])  if self._body_exists(rfoot_body)  else 0.0

        body_heights = np.array(
            [torso_z, pelvis_z, left_z, right_z],
            dtype=np.float32
        )

        # Foot contacts
        foot_contacts = self._get_foot_contacts()

        obs = np.concatenate([qpos, qvel, body_heights, foot_contacts], dtype=np.float32)
        return obs

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

    def _get_body_pos_feet(self) -> np.ndarray:
        """Return z-heights of both feet (or zeros if not found)."""
        lfoot = getattr(self, "_lfoot_body", "left_foot")
        rfoot = getattr(self, "_rfoot_body", "right_foot")
        left_z  = self._get_body_pos(lfoot)[2]  if self._body_exists(lfoot)  else 0.0
        right_z = self._get_body_pos(rfoot)[2] if self._body_exists(rfoot) else 0.0
        return np.array([left_z, right_z], dtype=np.float32)

    def _get_foot_contacts(self) -> np.ndarray:
        """Return binary foot contact indicators via contact force."""
        contacts = np.zeros(2, dtype=np.float32)
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            geom1_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            geom2_name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            if "left"  in (geom1_name + geom2_name).lower(): contacts[0] = 1.0
            if "right" in (geom1_name + geom2_name).lower(): contacts[1] = 1.0
        return contacts

    # ─── Reward ────────────────────────────────────────────────────────────

    def _get_reward(self, action: np.ndarray) -> float:
        state = self._build_state_dict(action)
        r  = self._loco_rewards.compute(state)
        r += self._stab_rewards.compute(state)
        r += self._energy_rewards.compute(state)
        return float(r)

    def _build_state_dict(self, action: np.ndarray) -> dict:
        """Build a state dict for reward functions."""
        d = self._data
        # Get torso body index for orientation matrix
        torso_body = getattr(self, "_torso_body", "torso")
        try:
            torso_bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, torso_body)
            upright_vec = d.xmat[torso_bid].reshape(3, 3)[:, 2]  # z-col = up direction
        except Exception:
            upright_vec = np.array([0.0, 0.0, 1.0])
        return {
            "root_vel":     d.qvel[:3].copy(),
            "root_ang_vel": d.qvel[3:6].copy(),
            "torso_height": float(self._get_body_pos(torso_body)[2]),
            "upright_vec":  upright_vec,
            "torques":      d.actuator_force.copy(),
            "action":       action,
            "target_speed": self._target_speed,
            "foot_contacts": self._get_foot_contacts(),
            "alive":        not self._is_terminated(),
        }

    # ─── Termination ───────────────────────────────────────────────────────

    def _is_terminated(self) -> bool:
        """Terminate if robot falls (torso too low or NaN)."""
        torso_body = getattr(self, "_torso_body", "torso")
        torso_z = self._get_body_pos(torso_body)[2]
        min_height = self.robot_cfg.get("min_torso_height", 0.5)
        if np.isnan(torso_z) or torso_z < min_height:
            return True
        if np.any(np.isnan(self._data.qpos)):
            return True
        return False

    def _is_truncated(self) -> bool:
        return self._step_count >= self._max_steps

    def _get_info(self) -> dict[str, Any]:
        torso_body = getattr(self, "_torso_body", "torso")
        return {
            "torso_height":  float(self._get_body_pos(torso_body)[2]),
            "x_velocity":    float(self._data.qvel[0]),
            "y_velocity":    float(self._data.qvel[1]),
            "foot_contacts": self._get_foot_contacts().tolist(),
        }

    # ─── Physics step ──────────────────────────────────────────────────────

    def _step_simulation(self, action: np.ndarray) -> None:
        # Scale normalized action [-1, 1] → actual torques
        torques = action * self._torque_limits
        self._data.ctrl[:] = torques

        for _ in range(self.frame_skip):
            mujoco.mj_step(self._model, self._data)

    # ─── Reset ─────────────────────────────────────────────────────────────

    def _reset_model(self) -> np.ndarray:
        # Reset to initial pose with small random perturbation
        noise_scale = self.robot_cfg.get("reset_noise_scale", 0.01)

        self._data.qpos[:] = self._init_qpos + self.np_random.uniform(
            -noise_scale, noise_scale, size=self._model.nq
        )
        self._data.qvel[:] = self._init_qvel + self.np_random.uniform(
            -noise_scale * 0.1, noise_scale * 0.1, size=self._model.nv
        )

        mujoco.mj_forward(self._model, self._data)
        return self._get_obs()

    # ─── Rendering ─────────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_offscreen()

    def _render_human(self):
        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self._model, self._data)
        self._viewer.sync()

    def _render_offscreen(self) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self._model, self._height, self._width)
        self._renderer.update_scene(self._data)
        return self._renderer.render()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer = None
