"""
scripts/train_quick.py
Quick training script — Unitree H1 with PD control + terrain + domain rand.

Runs 200k steps as a smoke test. After convergence, use train.py for full training.

Usage:
    source venv/bin/activate
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_quick.py
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_quick.py --task stair --n-envs 16
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.walk_task  import WalkTask
from humanoid_rl.envs.mujoco.tasks.run_task   import RunTask
from humanoid_rl.envs.mujoco.tasks.stair_task import StairTask
from humanoid_rl.envs.mujoco.tasks.jump_task  import JumpTask
from humanoid_rl.algorithms.ppo.ppo import PPOTrainer


TASK_MAP = {
    "walk":  WalkTask,
    "run":   RunTask,
    "stair": StairTask,
    "jump":  JumpTask,
}

DOMAIN_RAND_CFG = {
    "mass_enabled":     True,
    "mass_ratio":       0.15,
    "friction_enabled": True,
    "friction_ratio":   0.25,
    "damping_enabled":  True,
    "damping_ratio":    0.15,
    "gear_enabled":     True,
    "gear_ratio":       0.08,
}

PUSH_RAND_CFG = {
    "push_prob":      0.003,   # 0.3% per step
    "push_force_max": 60.0,    # N
    "push_duration":  5,       # steps
}


def make_env_fn(task_name: str, domain_rand: bool, push_rand: bool):
    task_obj  = TASK_MAP[task_name]()
    task_cfg  = task_obj.to_cfg()
    terrain   = task_cfg.get("terrain", "flat")
    terrain_kw = task_cfg.get("terrain_kwargs", {})

    def env_fn():
        return H1Env(
            task_cfg        = task_cfg,
            terrain_type    = terrain,
            terrain_kwargs  = terrain_kw,
            domain_rand_cfg = DOMAIN_RAND_CFG if domain_rand else None,
            push_rand_cfg   = PUSH_RAND_CFG   if push_rand  else None,
            frame_skip      = 10,              # 50Hz control
        )
    return env_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",         default="walk", choices=list(TASK_MAP.keys()))
    parser.add_argument("--n-envs",       type=int, default=8)
    parser.add_argument("--total-steps",  type=int, default=200_000)
    parser.add_argument("--no-rand",      action="store_true", help="Disable domain rand")
    parser.add_argument("--no-push",      action="store_true", help="Disable push rand")
    parser.add_argument("--log-dir",      default="runs/h1")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Humanoid-RL — Unitree H1 Training")
    print(f"  Task:       {args.task}")
    print(f"  N_envs:     {args.n_envs}")
    print(f"  Steps:      {args.total_steps:,}")
    print(f"  Domain Rand: {'OFF' if args.no_rand else 'ON'}")
    print(f"  Push Rand:   {'OFF' if args.no_push else 'ON'}")
    print("=" * 60)

    device  = "cuda" if torch.cuda.is_available() else "cpu"
    env_fn  = make_env_fn(args.task, not args.no_rand, not args.no_push)

    ppo_cfg = {
        "n_envs":        args.n_envs,
        "n_steps":       512,
        "batch_size":    256,
        "n_epochs":      4,
        "gamma":         0.99,
        "gae_lambda":    0.95,
        "clip_coef":     0.2,
        "clip_value":    True,
        "value_coef":    0.5,
        "entropy_coef":  0.005,
        "lr":            3e-4,
        "max_grad_norm": 1.0,
        "lr_anneal":     True,
        "total_steps":   args.total_steps,
        "save_every":    10,
        "actor_hidden":  [256, 128],
        "critic_hidden": [256, 128],
    }

    trainer = PPOTrainer(
        env_fn  = env_fn,
        cfg     = ppo_cfg,
        device  = device,
        log_dir = f"{args.log_dir}/{args.task}",
    )
    trainer.train()


if __name__ == "__main__":
    main()
