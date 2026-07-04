"""
scripts/train_curriculum.py
Curriculum training cho Unitree H1 / generic humanoid.
Auto-progression: STAND → SLOW WALK → WALK → GAIT → ROUGH TERRAIN.

Usage:
    source venv/bin/activate
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_curriculum.py

    # Dùng generic humanoid thay vì H1:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_curriculum.py --robot humanoid

    # Resume từ checkpoint:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_curriculum.py \\
        --resume runs/curriculum/checkpoints/checkpoint_step_100.pt

    # Chỉ chạy từ phase cụ thể:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_curriculum.py --start-phase 2

    # Override curriculum (chạy 1 task duy nhất):
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_curriculum.py --single-task walk
"""

from __future__ import annotations

import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import torch

# --- Env factories ---
def make_h1_env(task_cfg, domain_rand=False, push_rand=False, gait_rewards=False):
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    DOMAIN_RAND_CFG = {
        "mass_enabled": True, "mass_ratio": 0.15,
        "friction_enabled": True, "friction_ratio": 0.25,
        "damping_enabled": True, "damping_ratio": 0.15,
        "gear_enabled": True, "gear_ratio": 0.10,
    }
    PUSH_RAND_CFG = {
        "push_prob": 0.005, "push_force_max": 80.0, "push_duration": 5,
    }
    def env_fn():
        return H1Env(
            task_cfg=task_cfg, terrain_type="flat",
            domain_rand_cfg=DOMAIN_RAND_CFG if domain_rand else None,
            push_rand_cfg=PUSH_RAND_CFG if push_rand else None,
            frame_skip=10, history_len=3,  # multi-frame obs
        )
    return env_fn

def make_humanoid_env(task_cfg, domain_rand=False, push_rand=False, gait_rewards=False):
    from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
    from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
    DOMAIN_RAND_CFG = {
        "mass_enabled": True, "mass_ratio": 0.10,
        "friction_enabled": True, "friction_ratio": 0.20,
        "damping_enabled": True, "damping_ratio": 0.10,
        "gear_enabled": True, "gear_ratio": 0.05,
    }
    PUSH_RAND_CFG = {
        "push_prob": 0.003, "push_force_max": 50.0, "push_duration": 5,
    }
    robot_cfg = get_humanoid_cfg()
    def env_fn():
        return MuJoCoHumanoidEnv(
            robot_cfg=robot_cfg, task_cfg=task_cfg, terrain_type="flat",
            domain_rand_cfg=DOMAIN_RAND_CFG if domain_rand else None,
            push_rand_cfg=PUSH_RAND_CFG if push_rand else None,
            frame_skip=5, history_len=3,
        )
    return env_fn


def create_task_cfg(phase: dict) -> dict:
    """Build task config dict from phase config."""
    task_name = phase["task"]

    if task_name == "stand":
        from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
        task = StandTask(max_steps=500)
        cfg = task.to_cfg()
    elif task_name == "walk":
        from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
        task = WalkTask(
            target_speed=phase["target_speed"],
            max_steps=1000,
            terrain=phase["terrain"],
        )
        cfg = task.to_cfg()
        # Enable/disable gait rewards
        if not phase["gait_rewards"]:
            cfg["rewards"].pop("gait", None)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return cfg


def train_curriculum():
    parser = argparse.ArgumentParser(
        description="Curriculum training for humanoid robots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--robot", default="h1", choices=["h1", "humanoid"])
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--start-phase", type=int, default=0)
    parser.add_argument("--single-task", default=None,
                        help="Run single task instead of full curriculum: stand / walk")
    parser.add_argument("--log-dir", default="runs/curriculum")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Curriculum
    if args.single_task:
        from humanoid_rl.envs.mujoco.curriculum import DEFAULT_CURRICULUM
        phases = [p for p in DEFAULT_CURRICULUM if p["task"] == args.single_task]
        if not phases:
            phases = [DEFAULT_CURRICULUM[0]]
    else:
        from humanoid_rl.envs.mujoco.curriculum import DEFAULT_CURRICULUM
        phases = DEFAULT_CURRICULUM

    from humanoid_rl.envs.mujoco.curriculum import CurriculumManager
    curriculum = CurriculumManager(phases)

    from humanoid_rl.algorithms.ppo.ppo import PPOTrainer

    print("=" * 60)
    print(f"  Curriculum Training — Robot: {args.robot}")
    print(f"  Device: {device}  |  n_envs: {args.n_envs}")
    print(f"  Log:    {args.log_dir}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    print(f"  Start Phase: {args.start_phase}")
    print("=" * 60)
    print(curriculum.get_summary())
    print("=" * 60)

    # Skip phases before start_phase
    for _ in range(args.start_phase):
        if curriculum.is_done:
            break
        curriculum._advance()

    # Training loop
    trainer = None
    t_start = time.time()

    while not curriculum.is_done:
        phase = curriculum.current_phase
        print(f"\n{'='*60}")
        print(f"  Phase {curriculum.phase_index}: {phase['name']}")
        print(f"  target_speed={phase['target_speed']}  terrain={phase['terrain']}")
        print(f"  domain_rand={phase['domain_rand']}  push_rand={phase['push_rand']}")
        print(f"  gait_rewards={phase['gait_rewards']}")
        print(f"{'='*60}")

        task_cfg = create_task_cfg(phase)

        # Create env factory
        if args.robot == "h1":
            env_fn = make_h1_env(
                task_cfg,
                domain_rand=phase["domain_rand"],
                push_rand=phase["push_rand"],
                gait_rewards=phase["gait_rewards"],
            )
        else:
            env_fn = make_humanoid_env(
                task_cfg,
                domain_rand=phase["domain_rand"],
                push_rand=phase["push_rand"],
                gait_rewards=phase["gait_rewards"],
            )

        # PPO config
        ppo_cfg = {
            "n_envs": args.n_envs,
            "n_steps": 512,
            "batch_size": 1024,
            "n_epochs": 5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "clip_value": True,
            "value_coef": 0.5,
            "entropy_coef": phase.get("entropy_coef", 0.01),
            "lr": phase["lr"],
            "max_grad_norm": 0.5,
            "lr_anneal": True,
            "total_steps": phase["total_steps"],
            "save_every": 20,
            "actor_hidden": [512, 256, 128],
            "critic_hidden": [512, 256, 128],
        }

        # Create trainer
        trainer = PPOTrainer(
            env_fn=env_fn,
            cfg=ppo_cfg,
            device=device,
            log_dir=str(Path(args.log_dir) / f"phase_{curriculum.phase_index}_{phase['name'].lower().replace(' ','_')}"),
        )

        # Resume checkpoint if provided
        if args.resume and curriculum.phase_index == args.start_phase:
            ckpt_path = Path(args.resume)
            if ckpt_path.exists():
                trainer.load(str(ckpt_path))
                print(f"  Resumed from {ckpt_path}")

        # Hook into trainer to check curriculum after each update
        original_log = trainer.logger.log
        def curriculum_aware_log(metrics, step):
            nonlocal curriculum, trainer, original_log
            original_log(metrics, step)
            # Check curriculum advance
            result = curriculum.step(metrics)
            if result is not None:
                # Phase transition — save checkpoint
                trainer.save(f"phase_{curriculum.phase_index-1}_done")
                print(f"  Phase {curriculum.phase_index-1} done. Saving checkpoint.")
        trainer.logger.log = curriculum_aware_log

        # Train
        trainer.train()

        # Save phase checkpoint
        trainer.save(f"phase_{curriculum.phase_index}_final")

        # Check if curriculum advanced during training
        # (if not, manually advance after training steps complete)
        if curriculum.current_phase == phase:
            print(f"  Phase {curriculum.phase_index}: steps completed, advancing...")
            curriculum._advance()

        # Clear resume so we don't reload on next phase
        args.resume = None

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  ✅ Curriculum training complete!")
    print(f"  Total time: {elapsed/3600:.1f}h")
    print(f"  Final checkpoint: {Path(args.log_dir) / 'checkpoints' / 'checkpoint_final.pt'}")
    print(f"{'='*60}")

    if trainer:
        trainer.save("final")
        trainer.envs.close()
        trainer.logger.close()


if __name__ == "__main__":
    train_curriculum()
