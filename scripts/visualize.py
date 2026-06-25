"""
scripts/visualize.py
Policy visualization: real-time MuJoCo viewer OR offscreen video recorder.

Usage:
    # Real-time viewer (requires display):
    python scripts/visualize.py --robot humanoid --task walk --mode viewer

    # Record video without display:
    python scripts/visualize.py --robot humanoid --task walk \\
        --mode video --output rollout.mp4 --n-episodes 3

    # With trained policy:
    python scripts/visualize.py --robot h1 --task walk \\
        --checkpoint runs/ppo/checkpoints/checkpoint_latest.pt \\
        --mode video --output h1_walk.mp4
"""

import sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco
import mujoco.viewer


def build_robot_task_cfg(robot: str, task: str):
    if robot == "humanoid":
        from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
        robot_cfg = get_humanoid_cfg()
    else:
        from humanoid_rl.robots.unitree_h1 import get_h1_cfg
        robot_cfg = get_h1_cfg()

    task_map = {
        "walk":  "humanoid_rl.envs.mujoco.tasks.walk_task.WalkTask",
        "run":   "humanoid_rl.envs.mujoco.tasks.run_task.RunTask",
        "jump":  "humanoid_rl.envs.mujoco.tasks.jump_task.JumpTask",
        "stair": "humanoid_rl.envs.mujoco.tasks.stair_task.StairTask",
    }
    import importlib
    mod, cls = task_map[task].rsplit(".", 1)
    task_obj = getattr(importlib.import_module(mod), cls)()
    return robot_cfg, task_obj.to_cfg()


def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int):
    """Load PPO actor-critic from checkpoint (reads architecture from saved cfg)."""
    import torch
    from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic
    from humanoid_rl.utils.checkpoint import Checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = Checkpoint("").load(checkpoint_path)
    # Read architecture from saved config (fall back to defaults)
    cfg = state.get("cfg", {})
    actor_hidden  = cfg.get("actor_hidden",  [512, 256, 128])
    critic_hidden = cfg.get("critic_hidden", [512, 256, 128])
    ac = ActorCritic(obs_dim, act_dim,
                     actor_hidden=actor_hidden,
                     critic_hidden=critic_hidden).to(device)
    ac.load_state_dict(state["ac_state"])
    ac.eval()
    return ac, device


def get_action(ac, obs, device):
    """Get deterministic action from policy."""
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action_mean = ac.actor(ac.normalize_obs(obs_t))
    return action_mean.squeeze(0).cpu().numpy()


def run_viewer(robot: str, task: str, checkpoint: str | None, n_episodes: int):
    """Launch interactive MuJoCo passive viewer."""
    robot_cfg, task_cfg = build_robot_task_cfg(robot, task)
    xml_path = robot_cfg["mjcf_path_absolute"]

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)
    nu    = model.nu

    # Load policy if provided
    ac = device = None
    if checkpoint:
        from scripts.benchmark import make_env_fn
        tmp = make_env_fn(robot, task)()
        obs_dim = tmp.observation_space.shape[0]
        tmp.close()
        ac, device = load_policy(checkpoint, obs_dim, nu)
        print(f"[Viewer] Policy loaded: {checkpoint}")

    print(f"[Viewer] Launching MuJoCo viewer — robot={robot} task={task}")
    print("[Viewer] Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as v:
        for ep in range(n_episodes):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            step = 0

            while v.is_running():
                if ac is not None:
                    # Build obs (simple version: qpos[1:] + qvel)
                    obs = np.concatenate([data.qpos[1:], data.qvel]).astype(np.float32)
                    # Pad to obs_dim if needed
                    if len(obs) < ac.obs_mean.shape[0]:
                        pad = np.zeros(ac.obs_mean.shape[0] - len(obs), dtype=np.float32)
                        obs = np.concatenate([obs, pad])
                    action = get_action(ac, obs, device)
                else:
                    action = np.random.uniform(-1, 1, nu).astype(np.float32)

                tl = model.actuator_forcerange[:, 1]
                torques = action * (tl if not np.all(tl == 0) else np.ones(nu) * 100.0)
                data.ctrl[:] = torques

                for _ in range(5):  # frame_skip
                    mujoco.mj_step(model, data)

                v.sync()
                step += 1
                if step >= task_cfg.get("max_steps", 1000):
                    break
            if not v.is_running():
                break
        print("[Viewer] Closed.")


def run_video(robot: str, task: str, checkpoint: str | None,
              n_episodes: int, output: str, width: int = 640, height: int = 480, fps: int = 50):
    """Render offscreen and save to MP4."""
    import imageio
    robot_cfg, task_cfg = build_robot_task_cfg(robot, task)
    xml_path = robot_cfg["mjcf_path_absolute"]

    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    nu       = model.nu

    ac = device = None
    if checkpoint:
        from scripts.benchmark import make_env_fn
        tmp = make_env_fn(robot, task)()
        obs_dim = tmp.observation_space.shape[0]
        tmp.close()
        ac, device = load_policy(checkpoint, obs_dim, nu)
        print(f"[Video] Policy loaded: {checkpoint}")

    print(f"[Video] Recording {n_episodes} episode(s) → {output}")
    all_frames = []

    for ep in range(n_episodes):
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        step = 0

        while step < task_cfg.get("max_steps", 500):
            if ac is not None:
                obs = np.concatenate([data.qpos[1:], data.qvel]).astype(np.float32)
                if len(obs) < ac.obs_mean.shape[0]:
                    obs = np.concatenate([obs, np.zeros(ac.obs_mean.shape[0] - len(obs))])
                action = get_action(ac, obs, device)
            else:
                action = np.random.uniform(-1, 1, nu).astype(np.float32)

            tl = model.actuator_forcerange[:, 1]
            data.ctrl[:] = action * (tl if not np.all(tl == 0) else np.ones(nu) * 100.0)

            for _ in range(5):
                mujoco.mj_step(model, data)

            renderer.update_scene(data)
            frame = renderer.render()
            all_frames.append(frame)
            step += 1

        print(f"  Episode {ep+1}: {step} steps, {len(all_frames)} frames total")

    imageio.mimwrite(output, all_frames, fps=fps)
    print(f"[Video] Saved: {output}  ({len(all_frames)} frames @ {fps}fps)")


def main():
    parser = argparse.ArgumentParser(description="Humanoid-RL visualizer")
    parser.add_argument("--robot",      default="humanoid", choices=["humanoid","h1"])
    parser.add_argument("--task",       default="walk",     choices=["walk","run","stair","jump"])
    parser.add_argument("--mode",       default="video",    choices=["viewer","video"])
    parser.add_argument("--checkpoint", default=None,       help="Path to .pt checkpoint")
    parser.add_argument("--n-episodes", type=int, default=2)
    parser.add_argument("--output",     default="rollout.mp4")
    parser.add_argument("--width",      type=int, default=640)
    parser.add_argument("--height",     type=int, default=480)
    parser.add_argument("--fps",        type=int, default=50)
    args = parser.parse_args()

    if args.mode == "viewer":
        run_viewer(args.robot, args.task, args.checkpoint, args.n_episodes)
    else:
        run_video(args.robot, args.task, args.checkpoint,
                  args.n_episodes, args.output,
                  args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
