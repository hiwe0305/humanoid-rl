"""
scripts/record_tasks.py
Record rollout videos for all H1 tasks (walk, run, stair, jump).
Uses smooth random policy to show terrain and environment setup is correct.

Usage:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py
    PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py --task stair --steps 300
    PYTHONPATH=. MUJOCO_GL=egl python scripts/record_tasks.py --all --output-dir videos/
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

# Camera azimuth / elevation / distance / lookat for each task
CAMERAS = {
    "walk":  (135, -20, 4.5, [1.0, 0.0, 0.9]),
    "run":   (135, -18, 5.5, [2.0, 0.0, 0.9]),
    "stair": (150, -22, 6.0, [3.0, 0.0, 0.7]),
    "jump":  (135, -18, 6.5, [3.0, 0.0, 0.9]),
}


def smooth_action(prev: np.ndarray, rng: np.random.Generator,
                  smoothing: float = 0.88, scale: float = 0.12) -> np.ndarray:
    """Smooth random actions using exponential moving average."""
    noise = rng.uniform(-scale, scale, prev.shape).astype(np.float32)
    return np.clip(smoothing * prev + (1 - smoothing) * noise, -1.0, 1.0)


def make_scene_option() -> mujoco.MjvOption:
    """Configure what to show in the rendered scene."""
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    return opt


def record_task(
    task_name:  str,
    n_steps:    int   = 500,
    width:      int   = 960,
    height:     int   = 540,
    fps:        int   = 50,
    output_dir: Path  = Path("videos"),
    seed:       int   = 42,
) -> Path:
    task_cls, terrain, terrain_kw = TASK_CFG[task_name]
    task_cfg = task_cls().to_cfg()
    az, el, dist, lookat = CAMERAS[task_name]

    print(f"\n[{task_name.upper():5s}] Creating env — terrain={terrain} ...")
    env = H1Env(
        task_cfg       = task_cfg,
        terrain_type   = terrain,
        terrain_kwargs = terrain_kw,
        frame_skip     = 10,
        width          = width,
        height         = height,
    )

    # MuJoCo 3.x Renderer API
    renderer = mujoco.Renderer(env.model, height, width)

    # Camera setup using MjvCamera
    cam = mujoco.MjvCamera()
    cam.type     = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth  = az
    cam.elevation = el
    cam.distance  = dist
    cam.lookat[:] = lookat

    scene_opt = make_scene_option()

    rng    = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    nu     = env.action_space.shape[0]
    action = np.zeros(nu, dtype=np.float32)

    frames     = []
    ep_rewards = []
    ep_r       = 0.0
    n_eps      = 0
    print(f"[{task_name.upper():5s}] Recording {n_steps} steps ...")

    for step in range(n_steps):
        action = smooth_action(action, rng)
        obs, reward, term, trunc, info = env.step(action)
        ep_r += reward

        # Track robot x-position with camera
        rx = float(env.data.qpos[0])
        cam.lookat[0] = rx + (lookat[0] - 1.0)

        # Render with custom camera
        renderer.update_scene(env.data, camera=cam, scene_option=scene_opt)
        frames.append(renderer.render())

        if term or trunc:
            ep_rewards.append(ep_r)
            n_eps += 1
            print(f"  ep={n_eps}  step={step+1}  reward={ep_r:.1f}  "
                  f"torso_h={info['torso_height']:.2f}m  vx={info['x_velocity']:.2f}m/s")
            ep_r = 0.0
            obs, _ = env.reset(seed=seed + n_eps)
            action  = np.zeros(nu, dtype=np.float32)

    env.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"h1_{task_name}.mp4"
    imageio.mimwrite(str(out_path), frames, fps=fps, quality=8)

    mean_r = np.mean(ep_rewards) if ep_rewards else ep_r
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[{task_name.upper():5s}] Saved: {out_path}  "
          f"({len(frames)} frames | {size_mb:.1f} MB | mean_reward={mean_r:.1f})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Record H1 task rollout videos")
    parser.add_argument("--task",       default=None, choices=list(TASK_CFG.keys()))
    parser.add_argument("--steps",      type=int, default=500)
    parser.add_argument("--width",      type=int, default=960)
    parser.add_argument("--height",     type=int, default=540)
    parser.add_argument("--fps",        type=int, default=50)
    parser.add_argument("--output-dir", default="videos")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    tasks      = [args.task] if args.task else list(TASK_CFG.keys())
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print(f"  Unitree H1 — Task Video Recording")
    print(f"  Tasks:      {tasks}")
    print(f"  Steps/task: {args.steps}  |  Resolution: {args.width}×{args.height}")
    print(f"  Output dir: {output_dir}/")
    print("=" * 60)

    saved = []
    for task in tasks:
        path = record_task(
            task_name  = task,
            n_steps    = args.steps,
            width      = args.width,
            height     = args.height,
            fps        = args.fps,
            output_dir = output_dir,
            seed       = args.seed,
        )
        saved.append(path)

    print("\n" + "=" * 60)
    print("  Videos saved:")
    for p in saved:
        print(f"    {p}  ({p.stat().st_size/1024/1024:.1f} MB)")
    print("=" * 60)


if __name__ == "__main__":
    main()
