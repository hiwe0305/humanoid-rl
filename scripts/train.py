"""
scripts/train.py
Main training entry point. Supports Hydra config overrides.

Usage:
    # Train MuJoCo built-in humanoid on walk task (default):
    python scripts/train.py

    # Train Unitree H1 on run task:
    python scripts/train.py robot=h1 task=run

    # Override hyperparams:
    python scripts/train.py algo.lr=1e-4 algo.n_envs=32 train.total_steps=5000000

    # Resume from checkpoint:
    python scripts/train.py resume=runs/ppo/checkpoints/checkpoint_latest.pt
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from humanoid_rl.algorithms.ppo.ppo import PPOTrainer


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("  Humanoid-RL Training")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    # Device
    device = "cuda" if (cfg.train.use_gpu and torch.cuda.is_available()) else "cpu"
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[Train] Using CPU")

    # Build env factory
    env_fn = _build_env_fn(cfg)

    # PPO config
    ppo_cfg = {
        "n_envs":        cfg.algo.n_envs,
        "n_steps":       cfg.algo.n_steps,
        "batch_size":    cfg.algo.batch_size,
        "n_epochs":      cfg.algo.n_epochs,
        "gamma":         cfg.algo.gamma,
        "gae_lambda":    cfg.algo.gae_lambda,
        "clip_coef":     cfg.algo.clip_coef,
        "value_coef":    cfg.algo.value_coef,
        "entropy_coef":  cfg.algo.entropy_coef,
        "lr":            cfg.algo.lr,
        "max_grad_norm": cfg.algo.max_grad_norm,
        "lr_anneal":     cfg.algo.lr_anneal,
        "total_steps":   cfg.train.total_steps,
        "save_every":    cfg.train.save_every,
        "actor_hidden":  list(cfg.algo.actor_hidden),
        "critic_hidden": list(cfg.algo.critic_hidden),
    }

    # Run directory (Hydra sets cwd)
    log_dir = str(Path.cwd() / "logs")

    trainer = PPOTrainer(
        env_fn   = env_fn,
        cfg      = ppo_cfg,
        device   = device,
        log_dir  = log_dir,
    )

    # Resume if requested
    if cfg.get("resume"):
        trainer.load(cfg.resume)

    trainer.train()


def _build_env_fn(cfg: DictConfig):
    """Build environment factory function from config."""
    robot_name = cfg.robot.name
    task_name  = cfg.task.name

    if robot_name == "humanoid":
        from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
        robot_cfg = get_humanoid_cfg()
    elif robot_name == "h1":
        from humanoid_rl.robots.unitree_h1 import get_h1_cfg
        robot_cfg = get_h1_cfg()
    else:
        raise ValueError(f"Unknown robot: {robot_name}")

    if task_name == "walk":
        from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
        task = WalkTask(target_speed=cfg.task.get("target_speed", 1.0))
    elif task_name == "run":
        from humanoid_rl.envs.mujoco.tasks.run_task import RunTask
        task = RunTask(target_speed=cfg.task.get("target_speed", 3.0))
    elif task_name == "stair":
        from humanoid_rl.envs.mujoco.tasks.stair_task import StairTask
        task = StairTask()
    elif task_name == "jump":
        from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask
        task = JumpTask()
    else:
        raise ValueError(f"Unknown task: {task_name}")

    task_cfg = task.to_cfg()

    def env_fn():
        from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv

        # Handle built-in humanoid (absolute path)
        if "mjcf_path_absolute" in robot_cfg and robot_cfg["mjcf_file"] is None:
            import mujoco
            abs_path = robot_cfg["mjcf_path_absolute"]
            rc = dict(robot_cfg)
            rc["mjcf_file"] = "_builtin_"

            class BuiltinEnv(MuJoCoHumanoidEnv):
                def __init__(self):
                    # Bypass MJCF file lookup
                    from humanoid_rl.envs.base_env import BaseHumanoidEnv
                    BaseHumanoidEnv.__init__(self, rc, task_cfg, None, cfg.env.frame_skip)
                    self._width  = 640
                    self._height = 480

                    self._model = mujoco.MjModel.from_xml_path(abs_path)
                    self._data  = mujoco.MjData(self._model)
                    self._sim_dt  = self._model.opt.timestep
                    self._ctrl_dt = self._sim_dt * cfg.env.frame_skip

                    from gymnasium import spaces
                    import numpy as np
                    obs_dim = self._compute_obs_dim()
                    act_dim = self._model.nu
                    self._observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                    )
                    self._action_space = spaces.Box(
                        low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32
                    )
                    self._torque_limits = self._model.actuator_forcerange[:, 1].copy()
                    import numpy as np
                    if np.all(self._torque_limits == 0):
                        self._torque_limits = np.ones(act_dim) * rc.get("default_torque_limit", 100.0)

                    from humanoid_rl.rewards.locomotion import LocomotionRewards
                    from humanoid_rl.rewards.stability import StabilityRewards
                    from humanoid_rl.rewards.energy import EnergyRewards
                    rwd = task_cfg.get("rewards", {})
                    self._loco_rewards   = LocomotionRewards(rwd.get("locomotion", {}))
                    self._stab_rewards   = StabilityRewards(rwd.get("stability", {}))
                    self._energy_rewards = EnergyRewards(rwd.get("energy", {}))
                    self._max_steps      = task_cfg.get("max_steps", 1000)
                    self._target_speed   = task_cfg.get("target_speed", 1.0)
                    self._renderer = None
                    self._viewer   = None
                    import numpy as np
                    self._init_qpos = self._data.qpos.copy()
                    self._init_qvel = self._data.qvel.copy()

            return BuiltinEnv()
        else:
            return MuJoCoHumanoidEnv(
                robot_cfg    = robot_cfg,
                task_cfg     = task_cfg,
                render_mode  = None,
                frame_skip   = cfg.env.frame_skip,
            )

    return env_fn


if __name__ == "__main__":
    main()
