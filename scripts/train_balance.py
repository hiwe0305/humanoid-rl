#!/usr/bin/env python3
"""
Train H1 humanoid to BALANCE with MuJoCo.

Usage:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_balance.py
    PYTHONPATH=. MUJOCO_GL=glfw python scripts/train_balance.py --render
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_balance.py --record-video
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_balance.py --resume runs/h1_balance/checkpoints/checkpoint_step_100.pt

Output:
    runs/h1_balance/checkpoints/
    runs/h1_balance/videos/
    runs/h1_balance/logs/
"""

from __future__ import annotations

import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

def make_env(task_cfg: dict, render_mode: str | None = None):
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    def env_fn():
        return H1Env(
            task_cfg=task_cfg, terrain_type="flat",
            domain_rand_cfg=None, push_rand_cfg=None,
            render_mode=render_mode, frame_skip=10, history_len=3,
        )
    return env_fn


def train_balance():
    parser = argparse.ArgumentParser(
        description="Train H1 humanoid to balance with MuJoCo",
    )
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--total-steps", type=int, default=3_000_000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--log-dir", default="runs/h1_balance")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
    task_cfg = StandTask().to_cfg()
    render_mode = "human" if args.render else None
    env_fn = make_env(task_cfg, render_mode)

    ppo_cfg = {
        "n_envs": args.n_envs, "n_steps": 512, "batch_size": 1024,
        "n_epochs": 5, "gamma": 0.99, "gae_lambda": 0.95,
        "clip_coef": 0.2, "clip_value": True, "value_coef": 0.5,
        "entropy_coef": 0.01, "lr": 3e-4, "max_grad_norm": 0.5,
        "lr_anneal": True, "total_steps": args.total_steps, "save_every": 20,
        "actor_hidden": [512, 256, 128], "critic_hidden": [512, 256, 128],
        "record_video": args.record_video, "record_every": 50, "robot_name": "h1",
    }

    from humanoid_rl.algorithms.ppo.ppo import PPOTrainer

    print("=" * 60)
    print(f"  H1 Balance Training")
    print(f"  Device:  {device}")
    print(f"  n_envs:  {args.n_envs}")
    print(f"  Steps:   {args.total_steps:,}")
    print(f"  Log:     {args.log_dir}")
    if args.record_video:
        print(f"  Video:   every 50 updates")
    print("=" * 60)

    trainer = PPOTrainer(env_fn=env_fn, cfg=ppo_cfg, device=device, log_dir=args.log_dir)

    if args.resume:
        ckpt = Path(args.resume)
        if ckpt.exists():
            trainer.load(str(ckpt))
            print(f"  Resumed from {ckpt}")

    t_start = time.time()
    trainer.train()
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Time:    {elapsed/60:.1f} min ({elapsed/3600:.1f} h)")
    print(f"  FPS:     {args.total_steps / elapsed:.0f}")
    print(f"  Final:   {Path(args.log_dir) / 'checkpoints' / 'checkpoint_final.pt'}")
    print(f"{'='*60}")

if __name__ == "__main__":
    train_balance()
