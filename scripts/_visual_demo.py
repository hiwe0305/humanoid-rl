"""
Visual demo: Record H1 robot in MuJoCo doing 4 scenarios:
1. Idle standing (zero action) — does the robot hold posture?
2. PD joint control — apply specific joint targets, does it respond?
3. CPG walking gait — sinusoidal pattern, does it produce forward motion?
4. All 4 tasks init — show robot in stand/walk/run/jump configs

Output: videos/h1_demo_all.mp4
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco
import imageio

from humanoid_rl.envs.mujoco.h1_env import (
    H1Env, H1_DEFAULT_QPOS, H1_ACTION_SCALE,
)
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
from humanoid_rl.envs.mujoco.tasks.run_task import RunTask
from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask


WIDTH, HEIGHT = 960, 540
FPS = 50


def make_renderer_cam(env, az=135, el=-20, dist=4.0, lookat=(0, 0, 0.9)):
    renderer = mujoco.Renderer(env.model, HEIGHT, WIDTH)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = az
    cam.elevation = el
    cam.distance = dist
    cam.lookat[:] = lookat
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
    return renderer, cam, opt


def render_frame(renderer, env, cam, opt):
    cam.lookat[0] = float(env.data.qpos[0])
    renderer.update_scene(env.data, camera=cam, scene_option=opt)
    return renderer.render().copy()


# ─── Scenario 1: Idle Standing ─────────────────────────────────────────────
def scenario_idle():
    print("[Scenario 1] Idle Standing — zero action, watch PD hold...")
    env = H1Env(StandTask().to_cfg(), frame_skip=10, width=WIDTH, height=HEIGHT)
    renderer, cam, opt = make_renderer_cam(env)
    
    obs, _ = env.reset(seed=0)
    frames = []
    
    for step in range(150):  # 3 seconds
        obs, r, term, trunc, info = env.step(np.zeros(19))
        frames.append(render_frame(renderer, env, cam, opt))
        if step % 50 == 0:
            tz = info["torso_height"]
            vx = info["x_velocity"]
            print(f"  step={step:3d} torso_z={tz:.3f}m vx={vx:+.3f}m/s reward={r:+.2f}")
        if term:
            print(f"  → Robot fell at step {step}")
            # Pad remaining frames
            for _ in range(150 - step - 1):
                frames.append(frames[-1])
            break
    
    renderer.close()
    env.close()
    return frames


# ─── Scenario 2: PD Joint Response ─────────────────────────────────────────
def scenario_pd_response():
    print("\n[Scenario 2] PD Joint Response — commanding specific joints...")
    env = H1Env(StandTask().to_cfg(), frame_skip=10, width=WIDTH, height=HEIGHT)
    renderer, cam, opt = make_renderer_cam(env)
    
    obs, _ = env.reset(seed=0)
    frames = []
    
    # Phase 1: Squat (bend knees)
    for step in range(50):
        action = np.zeros(19, dtype=np.float32)
        action[2] = -0.5   # left hip pitch (flex)
        action[3] = 0.8    # left knee (bend)
        action[4] = -0.3   # left ankle
        action[7] = -0.5   # right hip pitch
        action[8] = 0.8    # right knee
        action[9] = -0.3   # right ankle
        obs, r, term, trunc, info = env.step(action)
        frames.append(render_frame(renderer, env, cam, opt))
        if term: break
    print(f"  Squat: torso_z={info['torso_height']:.3f}m (should be lower)")
    
    # Phase 2: Stand back up
    for step in range(50):
        obs, r, term, trunc, info = env.step(np.zeros(19))
        frames.append(render_frame(renderer, env, cam, opt))
        if term: break
    print(f"  Stand: torso_z={info['torso_height']:.3f}m (should recover)")
    
    # Phase 3: Arm raise
    for step in range(50):
        action = np.zeros(19, dtype=np.float32)
        action[11] = -0.8   # left shoulder pitch up
        action[15] = -0.8   # right shoulder pitch up
        obs, r, term, trunc, info = env.step(action)
        frames.append(render_frame(renderer, env, cam, opt))
        if term: break
    print(f"  Arms up: torso_z={info['torso_height']:.3f}m")
    
    renderer.close()
    env.close()
    return frames


# ─── Scenario 3: CPG Walking ───────────────────────────────────────────────
def scenario_walking():
    print("\n[Scenario 3] CPG Walking Gait — sinusoidal pattern...")
    env = H1Env(WalkTask(target_speed=1.0).to_cfg(), frame_skip=10, width=WIDTH, height=HEIGHT)
    renderer, cam, opt = make_renderer_cam(env, dist=5.0)
    ctrl_dt = env.model.opt.timestep * env.frame_skip
    
    obs, _ = env.reset(seed=0)
    frames = []
    phase = 0.0
    freq = 2.0
    max_vx = -999
    total_reward = 0
    
    for step in range(200):  # 4 seconds
        phase += ctrl_dt * freq * 2 * np.pi
        amp = 0.6
        action = np.zeros(19, dtype=np.float32)
        sin_l = np.sin(phase)
        sin_r = np.sin(phase + np.pi)
        
        action[2] = amp * sin_l * 0.8        # left hip pitch
        action[7] = amp * sin_r * 0.8        # right hip pitch
        action[3] = abs(sin_l) * amp * 0.6   # left knee
        action[8] = abs(sin_r) * amp * 0.6   # right knee
        action[4] = -sin_l * amp * 0.3       # left ankle
        action[9] = -sin_r * amp * 0.3       # right ankle
        action[11] = -amp * sin_l * 0.3      # left arm swing
        action[15] = -amp * sin_r * 0.3      # right arm swing
        
        obs, r, term, trunc, info = env.step(np.clip(action, -1, 1))
        frames.append(render_frame(renderer, env, cam, opt))
        total_reward += r
        vx = info["x_velocity"]
        max_vx = max(max_vx, vx)
        
        if step % 50 == 0:
            print(f"  step={step:3d} vx={vx:+.3f}m/s torso_z={info['torso_height']:.3f}m")
        
        if term:
            print(f"  → Robot fell at step {step}, max_vx={max_vx:.3f}m/s")
            obs, _ = env.reset(seed=step)
            phase = 0.0
    
    print(f"  Max forward velocity: {max_vx:.3f} m/s")
    print(f"  Total reward: {total_reward:.1f}")
    
    renderer.close()
    env.close()
    return frames


# ─── Scenario 4: All Tasks ─────────────────────────────────────────────────
def scenario_all_tasks():
    print("\n[Scenario 4] All Task Initialization...")
    all_frames = []
    
    tasks = [
        ("STAND", StandTask(), "flat"),
        ("WALK",  WalkTask(target_speed=1.0), "flat"),
        ("RUN",   RunTask(), "flat"),
        ("JUMP",  JumpTask(), "flat"),
    ]
    
    for name, task_obj, terrain in tasks:
        env = H1Env(task_obj.to_cfg(), terrain_type=terrain, frame_skip=10, width=WIDTH, height=HEIGHT)
        renderer, cam, opt = make_renderer_cam(env)
        obs, _ = env.reset(seed=42)
        
        # Show 40 frames each (0.8 sec)
        for step in range(40):
            obs, r, term, trunc, info = env.step(np.zeros(19))
            all_frames.append(render_frame(renderer, env, cam, opt))
            if term: break
        
        print(f"  {name:5s}: torso_z={info['torso_height']:.3f}m, reward={r:+.2f}")
        renderer.close()
        env.close()
    
    return all_frames


# ─── Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Unitree H1 — MuJoCo Environment Visual Demo")
    print("=" * 60)
    
    all_frames = []
    
    # Run all scenarios
    all_frames.extend(scenario_idle())
    all_frames.extend(scenario_pd_response())
    all_frames.extend(scenario_walking())
    all_frames.extend(scenario_all_tasks())
    
    # Save
    out = Path("videos/h1_demo_all.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), all_frames, fps=FPS, quality=8, macro_block_size=1)
    size_mb = out.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*60}")
    print(f"  ✅ Saved: {out} ({len(all_frames)} frames, {size_mb:.1f} MB)")
    print(f"  Duration: {len(all_frames)/FPS:.1f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
