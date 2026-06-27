#!/usr/bin/env python3
"""
scripts/view_mujoco.py
Mở MuJoCo Viewer tương tác để xem H1 robot.

Modes:
  1. zero_action:  H1 với zero action (test vật lý)
  2. policy:       H1 với trained policy (test inference)
  3. free:         H1 tĩnh để quan sát cấu trúc

Usage:
    # Xem robot với zero action (test đứng):
    PYTHONPATH=. python scripts/view_mujoco.py --mode zero_action

    # Xem robot với trained policy:
    PYTHONPATH=. python scripts/view_mujoco.py --mode policy \
        --checkpoint runs/h1_walk_full/checkpoints/checkpoint_final.pt

    # Chỉ xem mô hình tĩnh:
    PYTHONPATH=. python scripts/view_mujoco.py --mode free
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco
import mujoco.viewer

from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask


def run_zero_action(env: H1Env, handle):
    """Chạy zero action — test tính ổn định vật lý."""
    obs, _ = env.reset(seed=0)
    step = 0
    ep   = 0
    while handle.is_running():
        obs, r, term, trunc, info = env.step(np.zeros(env.action_space.shape[0]))
        handle.sync()
        step += 1
        if term or trunc:
            ep += 1
            print(f"[zero_action] ep={ep} survived={step} steps  "
                  f"torso={info['torso_height']:.3f}m  vx={info['x_velocity']:.3f}m/s")
            obs, _ = env.reset(seed=ep)
            step = 0
        time.sleep(0.02)   # 50 Hz realtime


def run_policy(env: H1Env, handle, checkpoint_path: str):
    """Chạy trained policy."""
    import torch
    from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg   = state.get("cfg", {})
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    ac = ActorCritic(
        obs_dim,
        act_dim,
        actor_hidden  = cfg.get("actor_hidden",  [512, 256, 128]),
        critic_hidden = cfg.get("critic_hidden", [512, 256, 128]),
    )
    ac.load_state_dict(state["ac_state"])
    ac.eval()
    print(f"[policy] Loaded  global_step={state.get('global_step',0):,}")

    obs, _ = env.reset(seed=0)
    step = ep = 0
    while handle.is_running():
        with torch.no_grad():
            obs_t  = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = ac.actor(ac.normalize_obs(obs_t)).clamp(-1.0, 1.0)
        action_np = action.squeeze(0).numpy()
        obs, r, term, trunc, info = env.step(action_np)
        handle.sync()
        step += 1
        if term or trunc:
            ep += 1
            print(f"[policy] ep={ep} survived={step}steps  "
                  f"torso={info['torso_height']:.3f}m  vx={info['x_velocity']:.3f}m/s  "
                  f"reward={r:.3f}")
            obs, _ = env.reset(seed=ep)
            step = 0
        time.sleep(0.02)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",       default="zero_action",
                        choices=["zero_action", "policy", "free"])
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    print(f"[viewer] mode={args.mode}")
    env = H1Env(WalkTask().to_cfg(), render_mode=None)
    env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data) as handle:
        # Camera setup — side view
        handle.cam.azimuth   = 135
        handle.cam.elevation = -15
        handle.cam.distance  = 4.0
        handle.cam.lookat[:] = [0.0, 0.0, 0.9]

        print("[viewer] MuJoCo viewer opened. Close window to exit.")
        print(f"         obs_dim={env.observation_space.shape[0]}  act_dim={env.action_space.shape[0]}")

        if args.mode == "zero_action":
            print("[viewer] Running ZERO ACTION — watch if robot stands or falls")
            run_zero_action(env, handle)
        elif args.mode == "policy":
            if not args.checkpoint:
                print("ERROR: --checkpoint required for policy mode")
                sys.exit(1)
            print(f"[viewer] Running TRAINED POLICY from {args.checkpoint}")
            run_policy(env, handle, args.checkpoint)
        else:  # free
            print("[viewer] FREE mode — robot static, use mouse to rotate view")
            while handle.is_running():
                handle.sync()
                time.sleep(0.016)

    env.close()
    print("[viewer] Closed.")


if __name__ == "__main__":
    main()
