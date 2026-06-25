"""
humanoid_rl/algorithms/ppo/actor_critic.py
Actor-Critic network for PPO.

Architecture:
  - Shared feature extractor (MLP)
  - Separate actor head (policy)
  - Separate critic head (value function)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def init_weights(module: nn.Module, gain: float = 1.0) -> None:
    """Orthogonal initialization (standard for PPO)."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm and ELU activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        output_gain: float = 1.0,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ELU())
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Initialize all linear layers
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                init_weights(m, gain=1.0)
        # Output layer: smaller gain for stability
        init_weights(self.net[-1], gain=output_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCritic(nn.Module):
    """
    Separate actor and critic networks.

    Actor: outputs mean of Gaussian policy (log_std learned as parameter)
    Critic: outputs scalar state value V(s)

    Args:
        obs_dim:        observation dimension
        act_dim:        action dimension
        actor_hidden:   hidden dims for actor
        critic_hidden:  hidden dims for critic
        log_std_init:   initial log standard deviation
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden: list[int] = [512, 256, 128],
        critic_hidden: list[int] = [512, 256, 128],
        log_std_init: float = -0.5,
    ):
        super().__init__()

        # Actor network (small output gain for initial near-zero actions)
        self.actor = MLP(obs_dim, actor_hidden, act_dim, output_gain=0.01)

        # Learned log standard deviation (shared across all dimensions)
        self.log_std = nn.Parameter(
            torch.ones(act_dim) * log_std_init
        )

        # Critic network
        self.critic = MLP(obs_dim, critic_hidden, 1, output_gain=1.0)

        # Observation normalization (running mean/std)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_var",  torch.ones(obs_dim))
        self.register_buffer("obs_count", torch.tensor(0.0))

    # ─── Observation normalization ────────────────────────────────────────

    @torch.no_grad()
    def update_obs_normalizer(self, obs: torch.Tensor) -> None:
        """Welford online algorithm for running mean/var."""
        batch_size = obs.shape[0]
        batch_mean = obs.mean(dim=0)
        batch_var  = obs.var(dim=0, unbiased=False)

        total = self.obs_count + batch_size
        delta = batch_mean - self.obs_mean
        new_mean = self.obs_mean + delta * (batch_size / total)
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_size
        new_var = (m_a + m_b + delta ** 2 * self.obs_count * batch_size / total) / total

        self.obs_mean  = new_mean
        self.obs_var   = new_var
        self.obs_count = total

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / (torch.sqrt(self.obs_var) + 1e-8)

    # ─── Forward ──────────────────────────────────────────────────────────

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute V(s). Shape: [B, 1]"""
        obs_norm = self.normalize_obs(obs)
        return self.critic(obs_norm)

    def get_action_and_value(
        self, obs: torch.Tensor, action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:   sampled or given action [B, act_dim]
            log_prob: log π(a|s) [B]
            entropy:  policy entropy [B]
            value:    V(s) [B, 1]
        """
        obs_norm = self.normalize_obs(obs)
        action_mean = self.actor(obs_norm)
        std = torch.exp(self.log_std).clamp(1e-3, 2.0)
        dist = Normal(action_mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        entropy  = dist.entropy().sum(dim=-1)
        value    = self.critic(obs_norm)

        return action, log_prob, entropy, value

    @torch.no_grad()
    def act(self, obs: np.ndarray, device: str = "cpu") -> tuple[np.ndarray, float, float]:
        """
        Inference: given obs numpy array, return (action, log_prob, value).
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        action, log_prob, _, value = self.get_action_and_value(obs_t)
        return (
            action.squeeze(0).cpu().numpy(),
            float(log_prob.item()),
            float(value.item()),
        )
