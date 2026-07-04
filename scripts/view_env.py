#!/usr/bin/env python3
"""
scripts/view_env.py
Xem humanoid trong MuJoCo environment — interactive viewer + video recording.

Usage:
    # Interactive viewer (cần display)
    MUJOCO_GL=glfw PYTHONPATH=. python scripts/view_env.py
    
    # Offscreen video recording (headless server)
    MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --record --steps 500
    
    # Dùng generic humanoid thay vì H1
    MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --robot humanoid --record
    
    # Load trained policy và render
    MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --checkpoint runs/curriculum/checkpoints/checkpoint_final.pt
    
    # Chạy stand task để verify PD control
    MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --task stand --record --steps 200
    
Modes:
    interactive:  bấm SPACE để reset, Q để thoát
    record:       tự động quay video ra output path
"""

from __future__ import annotations
import sys, argparse, time, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco

# ─── Model loading ──────────────────────────────────────────────────────────

def load_h1_model(terrain="flat"):
    """Load H1 model with terrain + offscreen buffer."""
    from robot_descriptions import h1_mj_description
    from humanoid_rl.envs.mujoco.terrain import inject_terrain
    xml_path = h1_mj_description.MJCF_PATH
    xml_dir = str(Path(xml_path).parent)
    with open(xml_path) as f:
        xml = f.read()
    xml = xml.replace('meshdir="assets"', f'meshdir="{xml_dir}/assets"')
    xml = xml.replace('texturedir="assets"', f'texturedir="{xml_dir}/assets"')
    xml = inject_terrain(xml, terrain)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data

def load_gen_model(terrain="flat"):
    """Load generic humanoid model."""
    from humanoid_rl.robots.mujoco_humanoid import HUMANOID_XML
    from humanoid_rl.envs.mujoco.terrain import inject_terrain
    xml = inject_terrain(HUMANOID_XML, terrain)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data

def reset_robot(model, data, robot="h1", task="stand"):
    """Reset robot to standing pose."""
    from humanoid_rl.envs.mujoco.h1_env import H1_DEFAULT_QPOS, H1_INIT_ROOT_Z, H1_INIT_QUAT
    from humanoid_rl.envs.mujoco.humanoid_loco import HUMANOID_DEFAULT_QPOS, HUMANOID_INIT_ROOT_Z, HUMANOID_INIT_QUAT
    
    mujoco.mj_resetData(model, data)
    if robot == "h1":
        data.qpos[2] = H1_INIT_ROOT_Z
        data.qpos[3:7] = H1_INIT_QUAT
        data.qpos[7:] = H1_DEFAULT_QPOS
    else:
        data.qpos[2] = HUMANOID_INIT_ROOT_Z
        data.qpos[3:7] = HUMANOID_INIT_QUAT
        data.qpos[7:] = HUMANOID_DEFAULT_QPOS
    mujoco.mj_forward(model, data)

# ─── Policy inference ───────────────────────────────────────────────────────

def load_policy(checkpoint_path, robot="h1"):
    """Load trained PPO checkpoint."""
    import torch
    from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic
    
    obs_dim = 82 * 3 if robot == "h1" else 76  # H1 uses history_len=3
    
    # Try to infer from checkpoint
    state = torch.load(checkpoint_path, map_location="cpu")
    ac_state = state.get("ac_state", state)
    
    # Determine act_dim from state dict
    actor_weight_shape = None
    for k, v in ac_state.items():
        if "actor.net.5.weight" in k or "actor.net.4.weight" in k:
            actor_weight_shape = v.shape
            break
    
    act_dim = actor_weight_shape[0] if actor_weight_shape else 19
    
    ac = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    ac.load_state_dict(ac_state)
    ac.eval()
    return ac

def run_policy(model, data, policy, device="cpu"):
    """Run one step of policy inference."""
    import torch
    # Build observation (simplified — just qpos + qvel)
    obs = np.concatenate([
        data.qpos[2:].flatten(),
        data.qvel.flatten(),
    ]).astype(np.float32)
    
    # Pad to expected dim if needed
    expected = policy.actor.net[0].in_features
    if len(obs) < expected:
        obs = np.pad(obs, (0, expected - len(obs)))
    
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy.actor(policy.normalize_obs(obs_t)).clamp(-1.0, 1.0)
    return action.squeeze(0).cpu().numpy()

# ─── Record video ───────────────────────────────────────────────────────────

def record_video(model, data, args):
    """Record video of robot behavior."""
    import cv2
    
    # Camera setup
    cam = mujoco.MjvCamera()
    cam.type = 0  # free
    cam.azimuth, cam.elevation, cam.distance = 135, -20, 4.5
    cam.lookat[:] = [0, 0, 0.9]
    
    renderer = mujoco.Renderer(model, 320, 480)
    frames = []
    
    # Load policy if checkpoint given
    policy = None
    if args.checkpoint:
        policy = load_policy(args.checkpoint, args.robot)
        print(f"[View] Loaded policy from {args.checkpoint}")
    
    # Load task config for reward computation
    task_fn = None
    if args.task == "stand":
        from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
        task_fn = StandTask()
    elif args.task == "walk":
        from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
        task_fn = WalkTask(target_speed=args.speed)
    
    # Main loop
    reset_robot(model, data, args.robot, args.task)
    
    for step in range(args.steps):
        # Get action
        if policy is not None:
            action = run_policy(model, data, policy)
        else:
            action = np.zeros(model.nu, dtype=np.float32)
        
        # Apply PD control if using H1
        if args.robot == "h1":
            from humanoid_rl.envs.mujoco.h1_env import H1_KP, H1_KD, H1_DEFAULT_QPOS, H1_TORQUE_LIMITS, H1_ACTION_SCALE
            kp = H1_KP; kd = H1_KD; torque_limit = H1_TORQUE_LIMITS; action_scale = H1_ACTION_SCALE
            q_target = H1_DEFAULT_QPOS + action * action_scale
        else:
            from humanoid_rl.envs.mujoco.humanoid_loco import HUMANOID_KP, HUMANOID_KD, HUMANOID_DEFAULT_QPOS, HUMANOID_ACTION_SCALE
            kp = HUMANOID_KP; kd = HUMANOID_KD; torque_limit = model.actuator_forcerange[:, 1]; action_scale = HUMANOID_ACTION_SCALE
            q_target = HUMANOID_DEFAULT_QPOS + action * action_scale
        
        torques = kp * (q_target - data.qpos[7:7+model.nu]) - kd * data.qvel[6:6+model.nu]
        torques = np.clip(torques, -torque_limit, torque_limit)
        data.ctrl[:] = torques
        
        # Step physics
        frame_skip = 10 if args.robot == "h1" else 5
        for _ in range(frame_skip):
            mujoco.mj_step(model, data)
        
        # Render
        cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.9]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())
        
        if (step + 1) % 100 == 0:
            torso_z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 
                "torso_link" if args.robot == "h1" else "torso"), 2]
            print(f"  step {step+1:5d}/{args.steps}  torso_z={torso_z:.3f}m  vx={data.qvel[0]:+.3f}m/s")
    
    renderer.close()
    
    # Write video
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(frames) > 0:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out), fourcc, 50.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        mb = out.stat().st_size / 1024 / 1024
        print(f"\n✅ Video saved: {out} ({len(frames)} frames, {mb:.1f} MB)")
    else:
        print("\n⚠️  No frames captured")

# ─── Interactive viewer ─────────────────────────────────────────────────────

def interactive_view(model, data, args):
    """Interactive MuJoCo viewer."""
    reset_robot(model, data, args.robot, args.task)
    
    # Keyboard state
    class KS:
        reset = False; quit = False
    
    def key_cb(keycode):
        if keycode == 32: KS.reset = True  # SPACE
        elif keycode in (81, 256): KS.quit = True  # Q/ESC
    
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 4.5
        viewer.cam.lookat[:] = [0, 0, 0.9]
        
        frame_skip = 10 if args.robot == "h1" else 5
        step = 0
        
        while viewer.is_running() and not KS.quit:
            if KS.reset:
                reset_robot(model, data, args.robot, args.task)
                KS.reset = False; step = 0
                print("[Reset]")
            
            # Zero action = PD holds default pose
            from humanoid_rl.envs.mujoco.h1_env import H1_KP, H1_KD, H1_DEFAULT_QPOS, H1_TORQUE_LIMITS, H1_ACTION_SCALE
            action = np.zeros(model.nu, dtype=np.float32)
            q_target = H1_DEFAULT_QPOS if args.robot == "h1" else np.zeros(model.nu)
            torques = H1_KP * (q_target - data.qpos[7:7+model.nu]) - H1_KD * data.qvel[6:6+model.nu]
            torques = np.clip(torques, -H1_TORQUE_LIMITS, H1_TORQUE_LIMITS)
            data.ctrl[:] = torques
            
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            step += 1
            
            viewer.cam.lookat[:] = [float(data.qpos[0]), float(data.qpos[1]), 0.9]
            
            if step % 200 == 0:
                torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 
                    "torso_link" if args.robot == "h1" else "torso")
                if torso_id >= 0:
                    tz = data.xpos[torso_id, 2]
                    print(f"  step={step:5d}  torso_z={tz:.3f}m  vx={data.qvel[0]:+.3f}m/s")
            
            viewer.sync()

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Humanoid RL Environment Viewer")
    parser.add_argument("--robot", default="h1", choices=["h1", "humanoid"])
    parser.add_argument("--task", default="stand", choices=["stand", "walk"])
    parser.add_argument("--speed", type=float, default=0.5, help="target speed (walk)")
    parser.add_argument("--record", action="store_true", help="record video instead of interactive")
    parser.add_argument("--output", default="output/humanoid_demo.mp4")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--checkpoint", default=None, help="load trained policy")
    parser.add_argument("--terrain", default="flat", choices=["flat", "rough", "stair", "slope", "hurdle", "gap"])
    args = parser.parse_args()
    
    print(f"[View] Loading {args.robot} robot, terrain={args.terrain}")
    
    if args.robot == "h1":
        model, data = load_h1_model(args.terrain)
    else:
        model, data = load_gen_model(args.terrain)
    
    print(f"[View] Model: nq={model.nq} nv={model.nv} nu={model.nu}")
    
    if args.record:
        record_video(model, data, args)
    else:
        interactive_view(model, data, args)

if __name__ == "__main__":
    main()
