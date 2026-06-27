"""
scripts/train_walk.py
Curriculum training cho Unitree H1 — 3 phases.

Phase 0 — STAND  (2M steps):  Học đứng vững, target_speed=0.0
Phase 1 — SLOW   (3M steps):  Học đi chậm,  target_speed=0.3 m/s
Phase 2 — WALK   (5M steps):  Học đi chuẩn, target_speed=1.0 m/s

Mỗi phase resume checkpoint từ phase trước.

Usage:
    source venv/bin/activate
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_walk.py

    # Resume từ checkpoint cụ thể (bỏ qua phase trước):
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_walk.py \\
        --skip-to-phase 1 \\
        --resume runs/h1_walk_full/checkpoints/checkpoint_final.pt

    # Chỉ chạy 1 phase:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_walk.py \\
        --only-phase 2 \\
        --resume runs/h1_walk_p1/checkpoints/checkpoint_final.pt
"""
from __future__ import annotations

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
from humanoid_rl.envs.mujoco.tasks.walk_task  import WalkTask
from humanoid_rl.algorithms.ppo.ppo import PPOTrainer


# ─── Domain Rand ────────────────────────────────────────────────────────────────
DOMAIN_RAND_CFG = {
    "mass_enabled":     True,  "mass_ratio":     0.10,
    "friction_enabled": True,  "friction_ratio": 0.20,
    "damping_enabled":  True,  "damping_ratio":  0.10,
    "gear_enabled":     True,  "gear_ratio":     0.05,
}
PUSH_RAND_CFG = {
    "push_prob": 0.002, "push_force_max": 50.0, "push_duration": 5,
}


# ─── Phase configs ───────────────────────────────────────────────────────────────
PHASE_CONFIGS = {
    0: {
        "name":        "STAND",
        "total_steps": 2_000_000,
        "task_fn":     lambda: StandTask().to_cfg(),
        "domain_rand": False,
        "push_rand":   False,
        "lr":          3e-4,
        "log_dir":     "runs/h1_phase0_stand",
    },
    1: {
        "name":        "SLOW WALK",
        "total_steps": 3_000_000,
        "task_fn":     lambda: WalkTask(target_speed=0.3, max_steps=1000).to_cfg(),
        "domain_rand": False,
        "push_rand":   False,
        "lr":          2e-4,
        "log_dir":     "runs/h1_phase1_slow",
    },
    2: {
        "name":        "WALK",
        "total_steps": 5_000_000,
        "task_fn":     lambda: WalkTask(target_speed=1.0, max_steps=1000).to_cfg(),
        "domain_rand": False,
        "push_rand":   False,
        "lr":          1e-4,
        "log_dir":     "runs/h1_phase2_walk",
    },
    3: {
        "name":        "GAIT",
        "total_steps": 5_000_000,
        "task_fn":     lambda: WalkTask(target_speed=1.0, max_steps=1000).to_cfg(),
        "domain_rand": False,
        "push_rand":   False,
        "lr":          5e-5,    # nhỏ hơn: fine-tune, không phá gait đã học
        "log_dir":     "runs/h1_phase3_gait",
    },
}


def make_env_fn(task_cfg: dict, domain_rand: bool = False, push_rand: bool = False):
    def env_fn():
        return H1Env(
            task_cfg        = task_cfg,
            terrain_type    = "flat",
            domain_rand_cfg = DOMAIN_RAND_CFG if domain_rand else None,
            push_rand_cfg   = PUSH_RAND_CFG   if push_rand  else None,
            frame_skip      = 10,
        )
    return env_fn


def train_phase(phase_id: int, n_envs: int, resume_from: str | None = None) -> str:
    cfg = PHASE_CONFIGS[phase_id]

    print(f"\n{'='*60}")
    print(f"  PHASE {phase_id} — {cfg['name']}")
    print(f"  Steps: {cfg['total_steps']:,}  |  n_envs: {n_envs}")
    print(f"  Log:   {cfg['log_dir']}")
    if resume_from:
        print(f"  Resume: {resume_from}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ppo_cfg = {
        "n_envs":           n_envs,
        "n_steps":          512,
        "batch_size":       1024,
        "n_epochs":         5,
        "gamma":            0.99,
        "gae_lambda":       0.95,
        "clip_coef":        0.2,
        "clip_value":       True,
        "value_coef":       0.5,
        "entropy_coef":     0.01,
        "lr":               cfg["lr"],
        "max_grad_norm":    0.5,
        "lr_anneal":        True,
        "normalize_obs":    True,
        "normalize_reward": False,   # raw rewards — trustworthy metrics
        "total_steps":      cfg["total_steps"],
        "save_every":       10,      # checkpoint mỗi 10 updates
        "actor_hidden":     [512, 256, 128],
        "critic_hidden":    [512, 256, 128],
    }

    task_cfg = cfg["task_fn"]()
    env_fn   = make_env_fn(task_cfg, domain_rand=cfg["domain_rand"], push_rand=cfg["push_rand"])

    trainer = PPOTrainer(
        env_fn  = env_fn,
        cfg     = ppo_cfg,
        device  = device,
        log_dir = cfg["log_dir"],
    )

    if resume_from and Path(resume_from).exists():
        trainer.load(resume_from)
        print(f"[Phase {phase_id}] Resumed from {resume_from}")
    elif resume_from:
        print(f"[Phase {phase_id}] WARNING: checkpoint not found: {resume_from}")

    trainer.train()

    final_ckpt = str(Path(cfg["log_dir"]) / "checkpoints" / "checkpoint_final.pt")
    print(f"[Phase {phase_id}] ✓ Saved: {final_ckpt}")
    return final_ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs",       type=int, default=16)
    parser.add_argument("--resume",       default=None,
                        help="Checkpoint để resume (cho phase đầu tiên chạy)")
    parser.add_argument("--skip-to-phase", type=int, default=0,
                        help="Bắt đầu từ phase này (0=stand, 1=slow, 2=walk)")
    parser.add_argument("--only-phase",   type=int, default=None,
                        help="Chỉ chạy 1 phase duy nhất")
    args = parser.parse_args()

    print("=" * 60)
    print("  Unitree H1 — Curriculum Walk Training")
    print("  Phase 0: STAND   (2M steps, target=0.0 m/s)")
    print("  Phase 1: SLOW    (3M steps, target=0.3 m/s)")
    print("  Phase 2: WALK    (5M steps, target=1.0 m/s)")
    print(f"  n_envs={args.n_envs}  |  start_phase={args.skip_to_phase}")
    print("=" * 60)

    # Xác định phases cần chạy
    if args.only_phase is not None:
        phases = [args.only_phase]
    else:
        phases = list(range(args.skip_to_phase, 3))

    prev_ckpt = args.resume

    for phase_id in phases:
        prev_ckpt = train_phase(
            phase_id    = phase_id,
            n_envs      = args.n_envs,
            resume_from = prev_ckpt,
        )

    print("\n✅ Curriculum training complete!")
    print(f"Final checkpoint: {prev_ckpt}")
    print()
    print("Render video:")
    print(f"  PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py \\")
    print(f"      --task walk --checkpoint {prev_ckpt} --steps 1500")


if __name__ == "__main__":
    main()
