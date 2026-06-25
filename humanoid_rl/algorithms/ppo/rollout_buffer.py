"""
humanoid_rl/algorithms/ppo/rollout_buffer.py
Python-side rollout buffer for PPO.

Stores N steps of experience from multiple parallel envs,
then computes advantages (GAE) for training.

Note: If C++ extension is built, the heavy collection loop
      uses humanoid_rl_cpp.RolloutBuffer for lock-free pushing;
      this class handles the final GAE computation.
"""

from __future__ import annotations

import numpy as np
import torch


class PPORolloutBuffer:
    """
    Stores trajectories for on-policy PPO updates.

    Buffers:
        observations  [n_steps, n_envs, obs_dim]
        actions       [n_steps, n_envs, act_dim]
        rewards       [n_steps, n_envs]
        dones         [n_steps, n_envs]
        values        [n_steps, n_envs]
        log_probs     [n_steps, n_envs]
        advantages    [n_steps, n_envs]  (computed via GAE)
        returns       [n_steps, n_envs]
    """

    def __init__(
        self,
        n_steps:  int,
        n_envs:   int,
        obs_dim:  int,
        act_dim:  int,
        gamma:    float = 0.99,
        gae_lambda: float = 0.95,
        device:   str = "cpu",
    ):
        self.n_steps    = n_steps
        self.n_envs     = n_envs
        self.obs_dim    = obs_dim
        self.act_dim    = act_dim
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = device

        self._ptr   = 0     # current step pointer
        self._full  = False

        self.observations = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions      = np.zeros((n_steps, n_envs, act_dim), dtype=np.float32)
        self.rewards      = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.dones        = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.values       = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.log_probs    = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.advantages   = np.zeros((n_steps, n_envs),          dtype=np.float32)
        self.returns      = np.zeros((n_steps, n_envs),          dtype=np.float32)

    # ─── Write ────────────────────────────────────────────────────────────

    def add(
        self,
        obs:      np.ndarray,  # [n_envs, obs_dim]
        action:   np.ndarray,  # [n_envs, act_dim]
        reward:   np.ndarray,  # [n_envs]
        done:     np.ndarray,  # [n_envs]
        value:    np.ndarray,  # [n_envs]
        log_prob: np.ndarray,  # [n_envs]
    ) -> None:
        self.observations[self._ptr] = obs
        self.actions[self._ptr]      = action
        self.rewards[self._ptr]      = reward
        self.dones[self._ptr]        = done
        self.values[self._ptr]       = value
        self.log_probs[self._ptr]    = log_prob

        self._ptr += 1
        if self._ptr >= self.n_steps:
            self._full = True
            self._ptr  = 0

    @property
    def is_full(self) -> bool:
        return self._full

    # ─── GAE ──────────────────────────────────────────────────────────────

    def compute_returns_and_advantages(
        self, last_values: np.ndarray, last_dones: np.ndarray
    ) -> None:
        """
        Generalized Advantage Estimation (GAE).

        Args:
            last_values: V(s_{T+1}) for each env [n_envs]
            last_dones:  done flag at T+1 for each env [n_envs]
        """
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                next_non_terminal = 1.0 - last_dones.astype(np.float32)
                next_values       = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values       = self.values[step + 1]

            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            self.advantages[step] = gae
            self.returns[step]    = gae + self.values[step]

    # ─── Batch generation ─────────────────────────────────────────────────

    def get_batches(self, batch_size: int):
        """
        Flatten buffer and yield random mini-batches.
        Normalizes advantages across entire buffer.

        Yields:
            dict with keys: obs, action, log_prob_old, advantage, return_, value_old
        """
        T, E = self.n_steps, self.n_envs
        n = T * E

        # Flatten
        obs       = self.observations.reshape(n, self.obs_dim)
        actions   = self.actions.reshape(n, self.act_dim)
        log_probs = self.log_probs.reshape(n)
        values    = self.values.reshape(n)
        advs      = self.advantages.reshape(n)
        returns   = self.returns.reshape(n)

        # Normalize advantages
        adv_mean = advs.mean()
        adv_std  = advs.std() + 1e-8
        advs     = (advs - adv_mean) / adv_std

        # Shuffle
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "obs":          torch.as_tensor(obs[batch_idx],       device=self.device),
                "action":       torch.as_tensor(actions[batch_idx],   device=self.device),
                "log_prob_old": torch.as_tensor(log_probs[batch_idx], device=self.device),
                "advantage":    torch.as_tensor(advs[batch_idx],      device=self.device),
                "return_":      torch.as_tensor(returns[batch_idx],   device=self.device),
                "value_old":    torch.as_tensor(values[batch_idx],    device=self.device),
            }

    def reset(self) -> None:
        self._ptr  = 0
        self._full = False
