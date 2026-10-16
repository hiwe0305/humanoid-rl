"""
humanoid_rl/algorithms/ppo/ppo.py
PPO trainer with:
  - Multiple parallel environments (subprocess or threaded)
  - Clipped surrogate objective
  - Value function loss with optional clipping
  - Entropy bonus
  - Gradient clipping
  - WandB + TensorBoard logging
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR

from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic
from humanoid_rl.algorithms.ppo.rollout_buffer import PPORolloutBuffer
from humanoid_rl.utils.logger import Logger
from humanoid_rl.utils.video_recorder import VideoRecorder
from humanoid_rl.utils.checkpoint import Checkpoint


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for humanoid locomotion.

    Args:
        env_fn:         callable that creates a single env instance
        cfg:            PPO hyperparameter dict
        device:         "cuda" or "cpu"
        log_dir:        directory for checkpoints and logs
    """

    DEFAULT_CFG = {
        # Rollout
        "n_envs":        16,
        "n_steps":       2048,
        "batch_size":    512,
        "n_epochs":      10,
        # Discount
        "gamma":         0.99,
        "gae_lambda":    0.95,
        # PPO clip
        "clip_coef":     0.2,
        "clip_value":    False,   # clip range is in return units; harmful with unnormalized returns
        # Losses
        "value_coef":    0.5,
        "entropy_coef":  0.005,
        # Optimization
        "lr":            3e-4,
        "max_grad_norm": 1.0,
        "lr_anneal":     True,
        # Total training
        "total_steps":   10_000_000,
        # Video recording
        "record_video":  False,   # enable periodic video recording
        "record_every":  50,      # record video every N updates
        "robot_name":    "h1",    # robot name for video recorder
        # Checkpointing
        "save_every":    100,    # save every N updates
        "eval_every":    50,
    }

    def __init__(
        self,
        env_fn,
        cfg: dict | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_dir: str = "runs/ppo",
        render_env_fn=None,
    ):
        self.cfg    = {**self.DEFAULT_CFG, **(cfg or {})}
        self.device = device
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create vectorized envs
        self.envs = self._make_vec_env(env_fn, self.cfg["n_envs"])

        # Get spaces from first env
        sample_env = env_fn()
        obs_dim = sample_env.observation_space.shape[0]
        act_dim = sample_env.action_space.shape[0]
        sample_env.close()

        # Actor-Critic
        self.ac = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            actor_hidden=self.cfg.get("actor_hidden",  [512, 256, 128]),
            critic_hidden=self.cfg.get("critic_hidden", [512, 256, 128]),
        ).to(device)

        print(f"[PPO] Actor-Critic parameters: {sum(p.numel() for p in self.ac.parameters()):,}")
        print(f"[PPO] obs_dim={obs_dim}  act_dim={act_dim}  device={device}")

        # Optimizer
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.cfg["lr"], eps=1e-5)

        # LR scheduler (linear annealing to 0)
        total_updates = self.cfg["total_steps"] // (self.cfg["n_steps"] * self.cfg["n_envs"])
        self.scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates
        ) if self.cfg["lr_anneal"] else None

        # Rollout buffer
        self.buffer = PPORolloutBuffer(
            n_steps    = self.cfg["n_steps"],
            n_envs     = self.cfg["n_envs"],
            obs_dim    = obs_dim,
            act_dim    = act_dim,
            gamma      = self.cfg["gamma"],
            gae_lambda = self.cfg["gae_lambda"],
            device     = device,
        )

        # Logger
        self.logger     = Logger(log_dir)
        self.checkpoint = Checkpoint(str(self.log_dir / "checkpoints"))

        # Reward normalization (running std)
        self._normalize_reward = self.cfg.get("normalize_reward", False)
        self._ret_rms_mean = 0.0
        self._ret_rms_var  = 1.0
        self._ret_rms_count = 0
        self._ret_running   = np.zeros(self.cfg["n_envs"], dtype=np.float32)  # discounted return tracker

        # Video recording (optional)
        self.video_recorder = None
        if self.cfg.get("record_video", False):
            self.video_recorder = VideoRecorder(
                env_fn=env_fn,
                log_dir=str(self.log_dir),
                record_every=self.cfg.get("record_every", 50),
                fps=50,
                robot_name=self.cfg.get("robot_name", "h1"),
            )
            print(f"[PPO] Video recording enabled: every {self.cfg.get('record_every', 50)} updates")

        # Interactive MuJoCo viewer (optional). Renders env 0 in real-time.
        self._viewer_env = None
        if render_env_fn is not None:
            self._viewer_env = render_env_fn()
            self._viewer_action = None
            print(f"[PPO] Interactive viewer enabled")

        # State
        self.global_step    = 0
        self.update_count   = 0
        self._obs           = None   # current obs [n_envs, obs_dim]
        self._dones         = None   # current done flags [n_envs]

    # ─── Env helpers ───────────────────────────────────────────────────────

    def _make_vec_env(self, env_fn, n_envs: int):
        """Create a simple synchronous vectorized env."""
        from humanoid_rl.utils.vec_env_wrapper import SyncVecEnv
        return SyncVecEnv([env_fn for _ in range(n_envs)])

    # ─── Training ──────────────────────────────────────────────────────────

    def train(self) -> None:
        """Main training loop."""
        total_steps  = self.cfg["total_steps"]
        n_envs       = self.cfg["n_envs"]
        n_steps      = self.cfg["n_steps"]
        steps_per_update = n_steps * n_envs

        # Initial reset
        obs_arr, _ = self.envs.reset()
        self._obs   = obs_arr.astype(np.float32)
        self._dones = np.zeros(n_envs, dtype=np.float32)

        print(f"[PPO] Starting training: total_steps={total_steps:,}")
        print(f"[PPO] steps_per_update={steps_per_update:,}  n_envs={n_envs}")
        t_start = time.time()

        # Support resuming: count steps from 0 within this training run
        # so `total_steps` is always interpreted as additional steps to train.
        steps_trained_this_run = 0

        while steps_trained_this_run < total_steps:
            # ── Collect rollout ───────────────────────────────────────────
            t_collect = time.time()
            self._collect_rollout()
            collect_time = time.time() - t_collect

            # ── Update policy ─────────────────────────────────────────────
            t_update = time.time()
            metrics = self._update_policy()
            update_time = time.time() - t_update

            self.update_count += 1
            self.global_step         += steps_per_update
            steps_trained_this_run   += steps_per_update

            # ── Logging ───────────────────────────────────────────────────
            fps = steps_per_update / (collect_time + update_time)
            metrics.update({
                "fps":          fps,
                "global_step":  self.global_step,
                "collect_time": collect_time,
                "update_time":  update_time,
            })
            self.logger.log(metrics, step=self.global_step)
            
            # Record training video
            if self.video_recorder is not None:
                self.video_recorder.step(
                    update_count=self.update_count,
                    metrics=metrics,
                )

            if self.update_count % 10 == 0:
                elapsed = time.time() - t_start
                print(
                    f"[PPO] step={self.global_step:>10,}  "
                    f"fps={fps:>6.0f}  "
                    f"loss_pi={metrics.get('loss_pi', 0):.4f}  "
                    f"loss_v={metrics.get('loss_v', 0):.4f}  "
                    f"ep_ret={metrics.get('mean_ep_return', 0):.2f}  "
                    f"ep_len={metrics.get('mean_ep_len', 0):.0f}  "
                    f"time={elapsed:.0f}s"
                )

            # ── Checkpoint ────────────────────────────────────────────────
            if self.update_count % self.cfg["save_every"] == 0:
                self.save(f"step_{self.global_step}")

            if self.scheduler is not None:
                self.scheduler.step()

        self.save("final")
        self.envs.close()
        if self._viewer_env is not None:
            self._viewer_env.close()
        self.logger.close()
        if self.video_recorder:
            self.video_recorder.close()
        print("[PPO] Training complete.")

    # ─── Rollout collection ────────────────────────────────────────────────

    def _collect_rollout(self) -> None:
        self.buffer.reset()
        self.ac.eval()

        ep_returns: list[float] = []
        ep_lengths: list[int] = []

        for _ in range(self.cfg["n_steps"]):
            with torch.no_grad():
                obs_t = torch.as_tensor(self._obs, device=self.device)
                self.ac.update_obs_normalizer(obs_t)
                actions, log_probs, _, values = self.ac.get_action_and_value(obs_t)

            actions_np  = actions.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()
            values_np   = values.squeeze(-1).cpu().numpy()

            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions_np)
            dones = (terminated | truncated).astype(np.float32)

            # Step interactive viewer to match env 0 (if enabled)
            if self._viewer_env is not None:
                if self._viewer_action is not None:
                    self._viewer_env.step(self._viewer_action)
                    self._viewer_env.render()
                self._viewer_action = actions_np[0].copy()

            # Reward normalization via running return std (Welford)
            if self._normalize_reward:
                gamma = self.cfg["gamma"]
                self._ret_running = self._ret_running * gamma + rewards
                # Update running stats (Chan et al. parallel variance)
                batch = self._ret_running
                n = len(batch)
                old_count = self._ret_rms_count
                tot = old_count + n
                delta = float(batch.mean()) - self._ret_rms_mean
                m2 = (self._ret_rms_var * old_count + float(np.var(batch)) * n
                      + delta ** 2 * old_count * n / max(tot, 1))
                self._ret_rms_mean += delta * n / tot
                self._ret_rms_var   = m2 / tot
                self._ret_rms_count = tot
                ret_std = float(np.sqrt(self._ret_rms_var + 1e-8))
                rewards = rewards / ret_std
                # Reset running return on episode end
                self._ret_running[dones > 0] = 0.0

            # Collect episode returns / lengths
            for i, (done, info) in enumerate(zip(dones, infos)):
                if done and "episode_return" in info:
                    ep_returns.append(info["episode_return"])
                    ep_lengths.append(info.get("episode_length", 0))

            # Time-limit truncation is NOT a failure: bootstrap with V(s_terminal)
            # so the critic is not taught that the future is worth 0 at the limit.
            trunc_only = truncated & ~terminated
            if trunc_only.any():
                term_obs = np.stack([
                    infos[i]["terminal_obs"] for i in np.where(trunc_only)[0]
                ]).astype(np.float32)
                with torch.no_grad():
                    v_term = self.ac.get_value(
                        torch.as_tensor(term_obs, device=self.device)
                    ).squeeze(-1).cpu().numpy()
                rewards = rewards.copy()
                rewards[trunc_only] += self.cfg["gamma"] * v_term

            self.buffer.add(
                obs      = self._obs,
                action   = actions_np,
                reward   = rewards.astype(np.float32),
                done     = dones,
                value    = values_np,
                log_prob = log_probs_np,
            )

            self._obs   = next_obs.astype(np.float32)
            self._dones = dones

        # Bootstrap value for last step
        with torch.no_grad():
            last_values = self.ac.get_value(
                torch.as_tensor(self._obs, device=self.device)
            ).squeeze(-1).cpu().numpy()

        self.buffer.compute_returns_and_advantages(last_values, self._dones)
        self._ep_returns = ep_returns
        self._ep_lengths = ep_lengths

    # ─── Policy update ────────────────────────────────────────────────────

    def _update_policy(self) -> dict[str, float]:
        self.ac.train()

        pi_losses, v_losses, entropy_losses, approx_kls, clip_fracs = [], [], [], [], []
        clip_coef   = self.cfg["clip_coef"]
        value_coef  = self.cfg["value_coef"]
        entropy_coef = self.cfg["entropy_coef"]
        max_grad    = self.cfg["max_grad_norm"]

        for _ in range(self.cfg["n_epochs"]):
            for batch in self.buffer.get_batches(self.cfg["batch_size"]):
                obs        = batch["obs"]
                action     = batch["action"]
                log_prob_old = batch["log_prob_old"]
                advantage  = batch["advantage"]
                return_    = batch["return_"]
                value_old  = batch["value_old"]

                _, log_prob_new, entropy, value_new = self.ac.get_action_and_value(obs, action)
                value_new = value_new.squeeze(-1)

                # Policy loss (clipped surrogate)
                ratio     = torch.exp(log_prob_new - log_prob_old)
                pg_loss1  = -advantage * ratio
                pg_loss2  = -advantage * ratio.clamp(1 - clip_coef, 1 + clip_coef)
                loss_pi   = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (with optional clipping)
                if self.cfg["clip_value"]:
                    v_clipped = value_old + (value_new - value_old).clamp(-clip_coef, clip_coef)
                    loss_v = torch.max(
                        (value_new - return_) ** 2,
                        (v_clipped  - return_) ** 2,
                    ).mean()
                else:
                    loss_v = ((value_new - return_) ** 2).mean()

                loss_ent  = entropy.mean()
                loss_total = loss_pi + value_coef * loss_v - entropy_coef * loss_ent

                self.optimizer.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_grad)
                self.optimizer.step()

                # Stats
                pi_losses.append(loss_pi.item())
                v_losses.append(loss_v.item())
                entropy_losses.append(loss_ent.item())
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                    clip_frac = (torch.abs(ratio - 1) > clip_coef).float().mean().item()
                approx_kls.append(approx_kl)
                clip_fracs.append(clip_frac)

        metrics = {
            "loss_pi":          np.mean(pi_losses),
            "loss_v":           np.mean(v_losses),
            "loss_entropy":     np.mean(entropy_losses),
            "approx_kl":        np.mean(approx_kls),
            "clip_fraction":    np.mean(clip_fracs),
            "learning_rate":    self.optimizer.param_groups[0]["lr"],
        }
        if self._ep_returns:
            metrics["mean_ep_return"] = np.mean(self._ep_returns)
            metrics["mean_ep_len"]    = np.mean(self._ep_lengths)
            metrics["n_episodes"]     = len(self._ep_returns)

        return metrics

    # ─── Save / Load ──────────────────────────────────────────────────────

    def save(self, tag: str = "latest") -> None:
        self.checkpoint.save({
            "ac_state":        self.ac.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step":     self.global_step,
            "update_count":    self.update_count,
            "cfg":             self.cfg,
        }, tag=tag)

    def load(self, path: str) -> None:
        state = self.checkpoint.load(path)
        src_dict = state["ac_state"]
        tgt_dict = self.ac.state_dict()

        src_in = src_dict["actor.net.0.weight"].shape[1]
        tgt_in = tgt_dict["actor.net.0.weight"].shape[1]

        if src_in == tgt_in:
            self.ac.load_state_dict(src_dict)
            try:
                self.optimizer.load_state_dict(state["optimizer_state"])
            except Exception as e:
                print(f"[PPO] Optimizer state could not be loaded ({e}), starting with fresh optimizer")
        elif src_in % 82 == 0 and tgt_in % 84 == 0 and (src_in // 82 == tgt_in // 84):
            # Transfer learning from no-gait model to gait model (e.g. 246 -> 252)
            h_len = src_in // 82
            new_dict = tgt_dict.copy()
            for k in ["actor.net.0.weight", "critic.net.0.weight"]:
                W_new = torch.zeros_like(new_dict[k])
                W_old = src_dict[k]
                for h in range(h_len):
                    W_new[:, h*84 : h*84+82] = W_old[:, h*82 : (h+1)*82]
                new_dict[k] = W_new

            for k in ["obs_mean", "obs_var"]:
                v_new = new_dict[k].clone()
                v_old = src_dict[k]
                for h in range(h_len):
                    v_new[h*84 : h*84+82] = v_old[h*82 : (h+1)*82]
                new_dict[k] = v_new

            for k, v in src_dict.items():
                if k not in ["actor.net.0.weight", "critic.net.0.weight", "obs_mean", "obs_var"]:
                    new_dict[k] = v
            self.ac.load_state_dict(new_dict)
            print(f"[PPO] Successfully transferred policy weights: obs_dim {src_in} -> {tgt_in} (preserving balance base features)")
        else:
            raise RuntimeError(f"Cannot load checkpoint with obs_dim {src_in} into model with obs_dim {tgt_in}")

        self.global_step  = state.get("global_step", 0)
        self.update_count = state.get("update_count", 0)
        print(f"[PPO] Loaded checkpoint: step={self.global_step:,}")
