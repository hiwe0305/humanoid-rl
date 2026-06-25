"""
humanoid_rl/envs/base_env.py
Abstract base class for all humanoid environments.
Defines the interface that all concrete envs must implement.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BaseHumanoidEnv(gym.Env, abc.ABC):
    """
    Abstract base for humanoid environments.

    Subclasses must implement:
        - observation_space (property)
        - action_space (property)
        - _get_obs() -> np.ndarray
        - _get_reward() -> float
        - _get_info() -> dict
        - _is_terminated() -> bool
        - _is_truncated() -> bool
        - _reset_model() -> np.ndarray  (reset and return first obs)
        - _step_simulation(action)       (apply action, step physics)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        robot_cfg: dict,
        task_cfg: dict,
        render_mode: str | None = None,
        frame_skip: int = 5,
    ):
        self.robot_cfg = robot_cfg
        self.task_cfg  = task_cfg
        self.render_mode = render_mode
        self.frame_skip  = frame_skip

        self._step_count = 0
        self._episode_return = 0.0

    # ─── Abstract interface ────────────────────────────────────────────────

    @property
    @abc.abstractmethod
    def observation_space(self) -> spaces.Box:
        ...

    @property
    @abc.abstractmethod
    def action_space(self) -> spaces.Box:
        ...

    @abc.abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return current observation."""
        ...

    @abc.abstractmethod
    def _get_reward(self, action: np.ndarray) -> float:
        """Compute reward for the current state after applying `action`."""
        ...

    @abc.abstractmethod
    def _get_info(self) -> dict[str, Any]:
        """Return auxiliary info dict."""
        ...

    @abc.abstractmethod
    def _is_terminated(self) -> bool:
        """True if episode ended due to failure (e.g. robot fell)."""
        ...

    @abc.abstractmethod
    def _is_truncated(self) -> bool:
        """True if episode ended due to time limit."""
        ...

    @abc.abstractmethod
    def _reset_model(self) -> np.ndarray:
        """Reset simulation state, return initial observation."""
        ...

    @abc.abstractmethod
    def _step_simulation(self, action: np.ndarray) -> None:
        """Apply action and advance physics by frame_skip steps."""
        ...

    # ─── Gymnasium interface ───────────────────────────────────────────────

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step_simulation(action)

        obs        = self._get_obs()
        reward     = self._get_reward(action)
        terminated = self._is_terminated()
        truncated  = self._is_truncated()
        info       = self._get_info()

        self._step_count += 1
        self._episode_return += reward
        info["step_count"]     = self._step_count
        info["episode_return"] = self._episode_return

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._step_count     = 0
        self._episode_return = 0.0

        obs = self._reset_model()
        info = self._get_info()
        info["step_count"]     = 0
        info["episode_return"] = 0.0

        if self.render_mode == "human":
            self.render()

        return obs, info

    def render(self):
        raise NotImplementedError(
            "Subclasses must implement render(). "
            "Use MuJoCoHumanoidEnv which provides render support."
        )

    def close(self):
        pass
