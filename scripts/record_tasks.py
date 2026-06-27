"""
scripts/record_tasks.py
Record rollout videos for all H1 tasks (walk, run, stair, jump).

Modes:
  - Random policy (default): verify env/terrain setup
  - Trained policy (--checkpoint): verify training quality

Usage:
    # Random policy (env check)
    PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py --task walk

    # Trained policy (inference video)
    PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py \
        --task walk --checkpoint runs/h1_walk_full/checkpoints/checkpoint_final.pt

    # All tasks, trained policy
    PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py \
        --checkpoint runs/h1_walk_full/checkpoints/checkpoint_final.pt
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco
import imageio

from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.walk_task  import WalkTask
from humanoid_rl.envs.mujoco.tasks.run_task   import RunTask
from humanoid_rl.envs.mujoco.tasks.stair_task import StairTask
from humanoid_rl.envs.mujoco.tasks.jump_task  import JumpTask


TASK_CFG = {
    "walk":  (WalkTask,  "flat",   {}),
    "run":   (RunTask,   "rough",  {"seed": 42}),
    "stair": (StairTask, "stair",  {"n_steps": 6, "step_height": 0.15, "step_depth": 0.40}),
    "jump":  (JumpTask,  "hurdle", {"n_hurdles": 4, "hurdle_height": 0.25, "spacing": 2.5}),
}

# Camera azimuth / elevation / distance / lookat
CAMERAS = {
    "walk":  (135, -20, 4.5, [1.0, 0.0, 0.9]),
    "run":   (135, -18, 5.5, [2.0, 0.0, 0.9]),
    "stair": (150, -22, 6.0, [3.0, 0.0, 0.7]),
    "jump":  (135, -18, 6.5, [3.0, 0.0, 0.9]),
}


# ─── Policy Loading ────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: str = "cpu"):
    """Load trained ActorCritic from a PPO checkpoint file."""
    import torch
    from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg   = state.get("cfg", {})
    ac = ActorCritic(
        obs_dim       = obs_dim,
        act_dim       = act_dim,
        actor_hidden  = cfg.get("actor_hidden",  [512, 256, 128]),
        critic_hidden = cfg.get("critic_hidden", [512, 256, 128]),
    ).to(device)
    ac.load_state_dict(state["ac_state"])
    ac.eval()
    step = state.get("global_step", 0)
    print(f"  [Policy] Loaded — global_step={step:,}  device={device}")
    return ac, step


def policy_action(ac, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Deterministic inference: mean action clipped to [-1, 1]."""
    import torch
    with torch.no_grad():
        obs_t    = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs_norm = ac.normalize_obs(obs_t)
        action   = ac.actor(obs_norm).clamp(-1.0, 1.0)   # standard PPO: clip at env boundary
    return action.squeeze(0).cpu().numpy()


# ─── Rendering helpers ─────────────────────────────────────────────────────────

def smooth_action(prev: np.ndarray, rng: np.random.Generator,
                  smoothing: float = 0.88, scale: float = 0.12) -> np.ndarray:
    noise = rng.uniform(-scale, scale, prev.shape).astype(np.float32)
    return np.clip(smoothing * prev + (1 - smoothing) * noise, -1.0, 1.0)


def make_scene_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]        = False
    return opt


def add_text_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    """Add a simple text overlay to a frame using numpy (no PIL dependency)."""
    # Simple pixel-level white block at bottom-left for text
    # (PIL/cv2 not required; text drawn by record loop prints to terminal)
    return frame


# ─── Main record function ──────────────────────────────────────────────────────

def record_task(
    task_name:       str,
    n_steps:         int        = 500,
    width:           int        = 960,
    height:          int        = 540,
    fps:             int        = 50,
    output_dir:      Path       = Path("videos"),
    seed:            int        = 42,
    checkpoint_path: str | None = None,
    device:          str        = "cpu",
    suffix:          str        = "",
) -> Path:
    """
    Record a video of H1 in the given task.

    Args:
        checkpoint_path: If provided, use trained policy; otherwise random policy.
        suffix:          Filename suffix (e.g. "_trained" or "_random")
    """
    task_cls, terrain, terrain_kw = TASK_CFG[task_name]
    task_cfg = task_cls().to_cfg()
    az, el, dist, lookat = CAMERAS[task_name]

    mode_str = "trained" if checkpoint_path else "random"
    print(f"\n[{task_name.upper():5s}] Creating env — terrain={terrain} | policy={mode_str}")
    env = H1Env(
        task_cfg       = task_cfg,
        terrain_type   = terrain,
        terrain_kwargs = terrain_kw,
        frame_skip     = 10,
        width          = width,
        height         = height,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load policy if checkpoint provided
    ac           = None
    global_step  = 0
    if checkpoint_path and Path(checkpoint_path).exists():
        ac, global_step = load_policy(checkpoint_path, obs_dim, act_dim, device)
    elif checkpoint_path:
        print(f"  [WARNING] Checkpoint not found: {checkpoint_path}  → falling back to random")

    renderer  = mujoco.Renderer(env.model, height, width)
    cam       = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth   = az
    cam.elevation = el
    cam.distance  = dist
    cam.lookat[:] = lookat
    scene_opt = make_scene_option()

    rng    = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    action = np.zeros(act_dim, dtype=np.float32)

    frames     = []
    ep_rewards = []
    ep_r       = 0.0
    n_eps      = 0
    ep_lengths = []
    ep_l       = 0

    print(f"[{task_name.upper():5s}] Recording {n_steps} steps ...")

    for step in range(n_steps):
        # Select action
        if ac is not None:
            action = policy_action(ac, obs, device=device)
        else:
            action = smooth_action(action, rng)

        obs, reward, term, trunc, info = env.step(action)
        ep_r += reward
        ep_l += 1

        # Track camera to robot
        rx = float(env.data.qpos[0])
        cam.lookat[0] = rx + (lookat[0] - 1.0)

        renderer.update_scene(env.data, camera=cam, scene_option=scene_opt)
        frames.append(renderer.render())

        if term or trunc:
            ep_rewards.append(ep_r)
            ep_lengths.append(ep_l)
            n_eps += 1
            print(f"  ep={n_eps}  len={ep_l}  reward={ep_r:.1f}  "
                  f"torso={info['torso_height']:.2f}m  "
                  f"vx={info['x_velocity']:.2f}m/s")
            ep_r = 0.0; ep_l = 0
            obs, _ = env.reset(seed=seed + n_eps)
            action  = np.zeros(act_dim, dtype=np.float32)

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    fname    = f"h1_{task_name}{suffix}.mp4"
    out_path = output_dir / fname
    imageio.mimwrite(str(out_path), frames, fps=fps, quality=8,
                     macro_block_size=1)   # avoid resize warning

    mean_r = np.mean(ep_rewards) if ep_rewards else ep_r
    mean_l = np.mean(ep_lengths) if ep_lengths else n_steps
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{task_name.upper():5s}] ✅ Saved: {out_path}")
    print(f"        {len(frames)} frames | {size_mb:.1f} MB | "
          f"mean_ep_len={mean_l:.0f} | mean_reward={mean_r:.1f}")
    return out_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Record H1 task rollout videos")
    parser.add_argument("--task",        default=None, choices=list(TASK_CFG.keys()),
                        help="Single task (default: all tasks)")
    parser.add_argument("--checkpoint",  default=None,
                        help="Path to trained checkpoint (.pt). If omitted, uses random policy.")
    parser.add_argument("--steps",       type=int, default=500)
    parser.add_argument("--width",       type=int, default=960)
    parser.add_argument("--height",      type=int, default=540)
    parser.add_argument("--fps",         type=int, default=50)
    parser.add_argument("--output-dir",  default="videos")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    tasks      = [args.task] if args.task else list(TASK_CFG.keys())
    output_dir = Path(args.output_dir)
    suffix     = "_trained" if args.checkpoint else "_random"

    print("=" * 60)
    print(f"  Unitree H1 — Task Video Recording")
    print(f"  Mode:       {'TRAINED POLICY' if args.checkpoint else 'RANDOM POLICY'}")
    print(f"  Tasks:      {tasks}")
    print(f"  Steps/task: {args.steps}  | Resolution: {args.width}×{args.height}")
    if args.checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    print("=" * 60)

    saved = []
    for task in tasks:
        path = record_task(
            task_name       = task,
            n_steps         = args.steps,
            width           = args.width,
            height          = args.height,
            fps             = args.fps,
            output_dir      = output_dir,
            seed            = args.seed,
            checkpoint_path = args.checkpoint,
            device          = args.device,
            suffix          = suffix,
        )
        saved.append(path)

    print("\n" + "=" * 60)
    print("  Videos saved:")
    for p in saved:
        print(f"    {p}  ({p.stat().st_size/1024/1024:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
