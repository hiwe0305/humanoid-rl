"""
scripts/train_balance.py
Train H1 humanoid to BALANCE (stand upright without falling).

This is the FOUNDATION for all locomotion tasks.
The trained checkpoint will be used as pretrained base for walk/run/jump.

Phase 1 — Basic Balance (3M steps):
    No disturbance, learn to hold upright posture with CoM over feet.

Phase 2 — Robust Balance (3M steps):
    Domain randomization + external push perturbation.
    Robot must recover from pushes and mass/friction variation.

Usage:
    source venv/bin/activate
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_balance.py

    # Resume:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_balance.py \\
        --resume runs/h1_balance_p1/checkpoints/checkpoint_final.pt \\
        --skip-to-phase 2

Output:
    runs/h1_balance_p2/checkpoints/checkpoint_final.pt
    → Use this as --resume for train_walk.py
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
from humanoid_rl.algorithms.ppo.ppo import PPOTrainer


# ─── Domain randomization ────────────────────────────────────────────────────
DOMAIN_RAND_CFG = {
    "mass_enabled":     True,  "mass_ratio":     0.15,   # ±15% mass
    "friction_enabled": True,  "friction_ratio": 0.25,   # ±25% friction
    "damping_enabled":  True,  "damping_ratio":  0.15,   # ±15% joint damping
    "gear_enabled":     True,  "gear_ratio":     0.10,   # ±10% actuator gear
}

PUSH_RAND_CFG = {
    "push_prob":       0.005,   # 0.5% chance per step → ~1 push per 4 seconds
    "push_force_max":  80.0,    # up to 80N horizontal push
    "push_duration":   5,       # 5 substeps
}


# ─── Phase configurations ────────────────────────────────────────────────────
PHASES = {
    1: {
        "name":        "BASIC BALANCE",
        "total_steps": 3_000_000,
        "domain_rand": False,
        "push_rand":   False,
        "lr":          3e-4,
        "log_dir":     "runs/h1_balance_p1",
    },
    2: {
        "name":        "ROBUST BALANCE",
        "total_steps": 3_000_000,
        "domain_rand": True,
        "push_rand":   True,
        "lr":          1e-4,
        "log_dir":     "runs/h1_balance_p2",
    },
}


def make_env_fn(domain_rand: bool = False, push_rand: bool = False):
    """Create H1 balance env factory."""
    def env_fn():
        task_cfg = StandTask().to_cfg()
        return H1Env(
            task_cfg        = task_cfg,
            terrain_type    = "flat",
            domain_rand_cfg = DOMAIN_RAND_CFG if domain_rand else None,
            push_rand_cfg   = PUSH_RAND_CFG   if push_rand  else None,
            frame_skip      = 10,
        )
    return env_fn


def train_phase(phase_id: int, n_envs: int, resume_from: str | None = None) -> str:
    """Train one phase of balance curriculum."""
    cfg = PHASES[phase_id]

    print(f"\n{'='*60}")
    print(f"  PHASE {phase_id} — {cfg['name']}")
    print(f"  Steps: {cfg['total_steps']:,}  |  n_envs: {n_envs}")
    print(f"  Domain rand: {cfg['domain_rand']}  |  Push rand: {cfg['push_rand']}")
    print(f"  Log: {cfg['log_dir']}")
    if resume_from:
        print(f"  Resume: {resume_from}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    ppo_cfg = {
        "n_envs":           n_envs,
        "n_steps":          512,       # rollout length
        "batch_size":       1024,
        "n_epochs":         5,
        "gamma":            0.99,
        "gae_lambda":       0.95,
        "clip_coef":        0.2,
        "clip_value":       True,
        "value_coef":       0.5,
        "entropy_coef":     0.01,      # encourage exploration
        "lr":               cfg["lr"],
        "max_grad_norm":    0.5,
        "lr_anneal":        True,
        "normalize_obs":    True,
        "normalize_reward": False,
        "total_steps":      cfg["total_steps"],
        "save_every":       10,        # checkpoint every 10 PPO updates
        "actor_hidden":     [512, 256, 128],
        "critic_hidden":    [512, 256, 128],
    }

    env_fn = make_env_fn(
        domain_rand=cfg["domain_rand"],
        push_rand=cfg["push_rand"],
    )

    trainer = PPOTrainer(
        env_fn  = env_fn,
        cfg     = ppo_cfg,
        device  = device,
        log_dir = cfg["log_dir"],
    )

    if resume_from and Path(resume_from).exists():
        trainer.load(resume_from)
        print(f"  [Phase {phase_id}] Resumed from {resume_from}")
    elif resume_from:
        print(f"  [Phase {phase_id}] WARNING: checkpoint not found: {resume_from}")

    trainer.train()

    final_ckpt = str(Path(cfg["log_dir"]) / "checkpoints" / "checkpoint_final.pt")
    print(f"\n  [Phase {phase_id}] ✅ Saved: {final_ckpt}")
    return final_ckpt


def main():
    parser = argparse.ArgumentParser(
        description="Train H1 balance policy (foundation for locomotion)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
After training, use the balance checkpoint as base for walk training:
  PYTHONPATH=. MUJOCO_GL=egl python scripts/train_walk.py \\
      --resume runs/h1_balance_p2/checkpoints/checkpoint_final.pt \\
      --skip-to-phase 1
""")
    parser.add_argument("--n-envs", type=int, default=16,
                        help="Number of parallel envs (default: 16)")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to resume from")
    parser.add_argument("--skip-to-phase", type=int, default=1,
                        help="Start from this phase (1=basic, 2=robust)")
    parser.add_argument("--only-phase", type=int, default=None,
                        help="Run only this phase")
    args = parser.parse_args()

    print("=" * 60)
    print("  Unitree H1 — Balance Training")
    print("  Phase 1: BASIC BALANCE  (3M steps, no disturbance)")
    print("  Phase 2: ROBUST BALANCE (3M steps, domain rand + push)")
    print(f"  n_envs={args.n_envs}")
    print("=" * 60)

    if args.only_phase is not None:
        phases = [args.only_phase]
    else:
        phases = list(range(args.skip_to_phase, 3))

    prev_ckpt = args.resume

    for phase_id in phases:
        if phase_id not in PHASES:
            print(f"[SKIP] Phase {phase_id} not defined")
            continue
        prev_ckpt = train_phase(
            phase_id    = phase_id,
            n_envs      = args.n_envs,
            resume_from = prev_ckpt,
        )

    print(f"\n{'='*60}")
    print(f"  ✅ Balance training complete!")
    print(f"  Final checkpoint: {prev_ckpt}")
    print()
    print("  Next: train walking on top of this balance policy:")
    print(f"    PYTHONPATH=. MUJOCO_GL=egl python scripts/train_walk.py \\")
    print(f"        --resume {prev_ckpt} --skip-to-phase 1")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
