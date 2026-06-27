"""
scripts/play_h1.py
Interactive real-time MuJoCo controller for Unitree H1.

Modes:
  manual  — Keyboard velocity commands → PD joint control
  policy  — Keyboard velocity commands → trained RL policy

Controls:
  W / ↑    Forward  (+0.5 m/s)
  S / ↓    Backward (-0.5 m/s)
  A / ←    Turn left
  D / →    Turn right
  SPACE    Reset robot
  R        Toggle recording
  Q / ESC  Quit

Usage:
    # Manual PD control (no trained model needed):
    PYTHONPATH=. python scripts/play_h1.py --mode manual

    # Policy inference (requires checkpoint):
    PYTHONPATH=. python scripts/play_h1.py --mode policy \\
        --checkpoint runs/h1_walk_full/checkpoints/checkpoint_final.pt

    # Record video while playing:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/play_h1.py --mode manual \\
        --record --output h1_play.mp4
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco
import mujoco.viewer

from humanoid_rl.envs.mujoco.h1_env import (
    H1Env, H1_DEFAULT_QPOS, H1_KP, H1_KD, H1_TORQUE_LIMITS,
    H1_INIT_ROOT_Z, H1_INIT_QUAT, H1_ACTION_SCALE,
)
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask


# ─── Keyboard State ──────────────────────────────────────────────────────────

class KeyboardState:
    """Track pressed keys for velocity command generation."""

    def __init__(self):
        self.forward  = False
        self.backward = False
        self.left     = False
        self.right    = False
        self.reset_requested = False
        self.quit_requested  = False
        self.recording       = False

    @property
    def cmd_vx(self) -> float:
        """Target forward velocity (m/s)."""
        vx = 0.0
        if self.forward:  vx += 1.0
        if self.backward: vx -= 0.5
        return vx

    @property
    def cmd_vy(self) -> float:
        """Target lateral velocity (m/s)."""
        return 0.0

    @property
    def cmd_yaw(self) -> float:
        """Target yaw rate (rad/s)."""
        yaw = 0.0
        if self.left:  yaw += 0.5
        if self.right: yaw -= 0.5
        return yaw


def key_callback(keycode: int, state: KeyboardState):
    """MuJoCo viewer key callback — updates keyboard state."""
    # MuJoCo key codes
    KEY_W     = 87
    KEY_S     = 83
    KEY_A     = 65
    KEY_D     = 68
    KEY_UP    = 265
    KEY_DOWN  = 267
    KEY_LEFT  = 263
    KEY_RIGHT = 262
    KEY_SPACE = 32
    KEY_R     = 82
    KEY_Q     = 81
    KEY_ESC   = 256

    if keycode == KEY_W or keycode == KEY_UP:
        state.forward = not state.forward
    elif keycode == KEY_S or keycode == KEY_DOWN:
        state.backward = not state.backward
    elif keycode == KEY_A or keycode == KEY_LEFT:
        state.left = not state.left
    elif keycode == KEY_D or keycode == KEY_RIGHT:
        state.right = not state.right
    elif keycode == KEY_SPACE:
        state.reset_requested = True
    elif keycode == KEY_R:
        state.recording = not state.recording
    elif keycode == KEY_Q or keycode == KEY_ESC:
        state.quit_requested = True


# ─── Manual PD Controller ────────────────────────────────────────────────────

class ManualController:
    """
    Generates joint targets from velocity commands using simple heuristics.

    This is NOT a proper walking controller — it's a manual tool for testing.
    For real walking, train an RL policy.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.nu    = model.nu
        self._phase = 0.0
        self._freq  = 2.0  # Hz — step frequency

        # PD controller targets (joint-space)
        self.q_target = H1_DEFAULT_QPOS.copy()

    def compute_action(self, data: mujoco.MjData,
                       cmd_vx: float, cmd_yaw: float,
                       dt: float) -> np.ndarray:
        """
        Convert velocity command to normalized action [-1, 1].

        Strategy:
        - cmd_vx = 0: hold default standing pose
        - cmd_vx > 0: generate simple walking pattern (sinusoidal leg swing)
        - cmd_yaw: add hip yaw offset for turning
        """
        action = np.zeros(self.nu, dtype=np.float32)

        if abs(cmd_vx) < 0.05 and abs(cmd_yaw) < 0.05:
            # Standing: zero action → PD holds default pose
            return action

        # Walking: generate a simple CPG-like pattern
        self._phase += dt * self._freq * 2 * np.pi
        phase = self._phase

        # Amplitude scales with speed
        amp = np.clip(abs(cmd_vx) * 0.5, 0.0, 0.8)
        sign = 1.0 if cmd_vx >= 0 else -1.0

        # Left leg (indices 0-4): hip_yaw, hip_roll, hip_pitch, knee, ankle
        # Right leg (indices 5-9): same, anti-phase
        sin_l = np.sin(phase)
        sin_r = np.sin(phase + np.pi)  # anti-phase

        # Hip pitch: swing legs forward/backward
        action[2] = sign * amp * sin_l * 0.8       # left hip_pitch
        action[7] = sign * amp * sin_r * 0.8       # right hip_pitch

        # Knee: bend during swing phase
        action[3] = abs(sin_l) * amp * 0.6          # left knee
        action[8] = abs(sin_r) * amp * 0.6          # right knee

        # Ankle: plantarflex during push-off
        action[4] = -sin_l * amp * 0.3              # left ankle
        action[9] = -sin_r * amp * 0.3              # right ankle

        # Yaw: turn by offsetting hip yaw
        if abs(cmd_yaw) > 0.05:
            yaw_offset = cmd_yaw * 0.3
            action[0] += yaw_offset   # left hip yaw
            action[5] -= yaw_offset   # right hip yaw

        # Arm swing (natural counter-rotation)
        action[11] = -sign * amp * sin_l * 0.3   # left shoulder pitch
        action[15] = -sign * amp * sin_r * 0.3   # right shoulder pitch

        return np.clip(action, -1.0, 1.0)


# ─── Policy Loader ────────────────────────────────────────────────────────────

def load_policy(checkpoint_path: str, obs_dim: int, act_dim: int, device: str = "cpu"):
    """Load trained ActorCritic from checkpoint."""
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

    global_step = state.get("global_step", 0)
    print(f"  [Policy] Loaded — global_step={global_step:,}  device={device}")
    return ac


def policy_action(ac, obs: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Get deterministic action from policy."""
    import torch
    with torch.no_grad():
        obs_t    = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs_norm = ac.normalize_obs(obs_t)
        action   = ac.actor(obs_norm).clamp(-1.0, 1.0)
    return action.squeeze(0).cpu().numpy()


# ─── Main Loop ────────────────────────────────────────────────────────────────

def run_interactive(
    mode:            str  = "manual",
    checkpoint_path: str  | None = None,
    record:          bool = False,
    output:          str  = "h1_play.mp4",
    max_steps:       int  = 10000,
    fps:             int  = 50,
):
    """Run interactive MuJoCo viewer with H1 robot."""

    # Create environment
    task_cfg = WalkTask(target_speed=1.0, max_steps=max_steps).to_cfg()
    env = H1Env(
        task_cfg     = task_cfg,
        terrain_type = "flat",
        frame_skip   = 10,
        width        = 960,
        height       = 540,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    print(f"{'='*60}")
    print(f"  Unitree H1 — Interactive Controller")
    print(f"  Mode: {mode.upper()}")
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}")
    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")
    print()
    print("  Controls:")
    print("    W/↑ = Forward    S/↓ = Backward")
    print("    A/← = Turn Left  D/→ = Turn Right")
    print("    SPACE = Reset    Q/ESC = Quit")
    if record:
        print("    R = Toggle Recording")
    print()

    # Load policy if needed
    ac = None
    device = "cpu"
    if mode == "policy":
        if checkpoint_path and Path(checkpoint_path).exists():
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ac = load_policy(checkpoint_path, obs_dim, act_dim, device)
        else:
            print("[WARNING] No valid checkpoint — falling back to manual mode")
            mode = "manual"

    # Manual controller
    manual_ctrl = ManualController(env.model)
    ctrl_dt = env.model.opt.timestep * env.frame_skip  # control dt

    # Keyboard state
    kb = KeyboardState()

    # Recording
    frames = []
    renderer = None
    if record:
        renderer = mujoco.Renderer(env.model, 540, 960)

    # Reset environment
    obs, info = env.reset(seed=42)
    total_reward = 0.0
    ep_count = 0
    step_count = 0

    print(f"[Play] Launching MuJoCo viewer...")
    print(f"[Play] Close window or press Q to exit.\n")

    with mujoco.viewer.launch_passive(
        env.model, env.data,
        key_callback=lambda keycode: key_callback(keycode, kb),
    ) as viewer:
        # Set initial camera
        viewer.cam.azimuth   = 135
        viewer.cam.elevation = -20
        viewer.cam.distance  = 4.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.9]

        t_start = time.time()
        t_step  = t_start

        while viewer.is_running() and not kb.quit_requested:
            # Handle reset
            if kb.reset_requested:
                obs, info = env.reset(seed=42 + ep_count)
                total_reward = 0.0
                step_count = 0
                ep_count += 1
                manual_ctrl._phase = 0.0
                kb.reset_requested = False
                print(f"[Reset] Episode {ep_count}")

            # Generate action
            if mode == "policy" and ac is not None:
                action = policy_action(ac, obs, device)
            else:
                action = manual_ctrl.compute_action(
                    env.data, kb.cmd_vx, kb.cmd_yaw, ctrl_dt)

            # Step environment
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            step_count += 1

            # Track camera to robot
            rx = float(env.data.qpos[0])
            ry = float(env.data.qpos[1])
            viewer.cam.lookat[0] = rx
            viewer.cam.lookat[1] = ry

            # Record frame if recording
            if record and kb.recording and renderer is not None:
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                cam.azimuth   = viewer.cam.azimuth
                cam.elevation = viewer.cam.elevation
                cam.distance  = viewer.cam.distance
                cam.lookat[:] = viewer.cam.lookat
                renderer.update_scene(env.data, camera=cam)
                frames.append(renderer.render().copy())

            # Print periodic status
            if step_count % 100 == 0:
                vx = info.get("x_velocity", 0.0)
                tz = info.get("torso_height", 0.0)
                cmd = f"cmd_vx={kb.cmd_vx:.1f}"
                rec_str = " [REC]" if kb.recording else ""
                print(f"  step={step_count:5d} vx={vx:+6.2f} torso_z={tz:.3f} "
                      f"reward={total_reward:+8.1f} {cmd}{rec_str}")

            # Auto-reset on termination
            if term or trunc:
                vx = info.get("x_velocity", 0.0)
                tz = info.get("torso_height", 0.0)
                print(f"  [DONE] step={step_count} torso_z={tz:.3f} "
                      f"vx={vx:.2f} total_r={total_reward:.1f}")
                obs, info = env.reset(seed=42 + ep_count)
                total_reward = 0.0
                step_count = 0
                ep_count += 1
                manual_ctrl._phase = 0.0

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            t_now = time.time()
            t_elapsed = t_now - t_step
            t_target = ctrl_dt
            if t_elapsed < t_target:
                time.sleep(t_target - t_elapsed)
            t_step = time.time()

    env.close()

    # Save recorded video
    if frames:
        import imageio
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out_path), frames, fps=fps, quality=8,
                         macro_block_size=1)
        print(f"\n[Play] Video saved: {out_path} ({len(frames)} frames)")

    elapsed = time.time() - t_start
    print(f"\n[Play] Session ended. {ep_count} episodes, {elapsed:.0f}s total.")


# ─── Offscreen Video Mode ────────────────────────────────────────────────────

def run_offscreen(
    mode:            str  = "manual",
    checkpoint_path: str  | None = None,
    output:          str  = "h1_play.mp4",
    n_steps:         int  = 1000,
    fps:             int  = 50,
    width:           int  = 960,
    height:          int  = 540,
):
    """Record offscreen video (no display needed)."""
    import imageio

    task_cfg = WalkTask(target_speed=1.0, max_steps=n_steps).to_cfg()
    env = H1Env(
        task_cfg     = task_cfg,
        terrain_type = "flat",
        frame_skip   = 10,
        width        = width,
        height       = height,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Load policy
    ac = None
    device = "cpu"
    if mode == "policy" and checkpoint_path and Path(checkpoint_path).exists():
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ac = load_policy(checkpoint_path, obs_dim, act_dim, device)

    renderer = mujoco.Renderer(env.model, height, width)
    cam = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth   = 135
    cam.elevation = -20
    cam.distance  = 4.0
    cam.lookat[:] = [0.0, 0.0, 0.9]

    obs, _ = env.reset(seed=42)
    ctrl_dt = env.model.opt.timestep * env.frame_skip

    # For manual mode: simulate walking forward
    manual_ctrl = ManualController(env.model)

    frames = []
    total_r = 0.0

    print(f"[Record] Recording {n_steps} steps → {output}")
    for step in range(n_steps):
        if ac is not None:
            action = policy_action(ac, obs, device)
        else:
            # Simulate forward walking command
            action = manual_ctrl.compute_action(env.data, 1.0, 0.0, ctrl_dt)

        obs, reward, term, trunc, info = env.step(action)
        total_r += reward

        # Track camera
        cam.lookat[0] = float(env.data.qpos[0])

        renderer.update_scene(env.data, camera=cam)
        frames.append(renderer.render().copy())

        if term or trunc:
            print(f"  [DONE] step={step} torso_z={info['torso_height']:.3f} "
                  f"vx={info['x_velocity']:.2f} reward={total_r:.1f}")
            obs, _ = env.reset(seed=42 + step)
            total_r = 0.0

    env.close()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out_path), frames, fps=fps, quality=8,
                     macro_block_size=1)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[Record] ✅ Saved: {out_path} ({len(frames)} frames, {size_mb:.1f} MB)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive real-time MuJoCo controller for Unitree H1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Controls (interactive mode):
  W/↑      Forward         S/↓      Backward
  A/←      Turn Left       D/→      Turn Right
  SPACE    Reset Robot     Q/ESC    Quit
  R        Toggle Record

Examples:
  # Manual control (interactive viewer):
  PYTHONPATH=. python scripts/play_h1.py --mode manual

  # Policy inference:
  PYTHONPATH=. python scripts/play_h1.py --mode policy --checkpoint <path.pt>

  # Record offscreen video (no display):
  PYTHONPATH=. MUJOCO_GL=egl python scripts/play_h1.py \\
      --mode manual --offscreen --output h1_walk.mp4
""")
    parser.add_argument("--mode", default="manual", choices=["manual", "policy"],
                        help="Control mode (default: manual)")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained .pt checkpoint (required for policy mode)")
    parser.add_argument("--offscreen", action="store_true",
                        help="Record offscreen video instead of interactive viewer")
    parser.add_argument("--record", action="store_true",
                        help="Enable recording in interactive mode (press R to toggle)")
    parser.add_argument("--output", default="h1_play.mp4",
                        help="Output video path (default: h1_play.mp4)")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Max steps (interactive) or total steps (offscreen)")
    parser.add_argument("--fps", type=int, default=50,
                        help="Video FPS (default: 50)")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    args = parser.parse_args()

    if args.offscreen:
        run_offscreen(
            mode            = args.mode,
            checkpoint_path = args.checkpoint,
            output          = args.output,
            n_steps         = args.steps,
            fps             = args.fps,
            width           = args.width,
            height          = args.height,
        )
    else:
        run_interactive(
            mode            = args.mode,
            checkpoint_path = args.checkpoint,
            record          = args.record,
            output          = args.output,
            max_steps       = args.steps,
            fps             = args.fps,
        )


if __name__ == "__main__":
    main()
