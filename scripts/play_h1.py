"""
scripts/play_h1.py
Interactive real-time MuJoCo controller for Unitree H1.

Uses direct MuJoCo control with:
- PD joint position control
- Virtual CoM stabilization force (simulates ZMP balance control)
- CPG walking pattern

Controls:
  W / ↑    Walk forward        S / ↓    Walk backward
  A / ←    Turn left            D / →    Turn right
  SPACE    Reset robot          Q / ESC  Quit

Usage:
    PYTHONPATH=. MUJOCO_GL=glfw python scripts/play_h1.py
    PYTHONPATH=. MUJOCO_GL=egl  python scripts/play_h1.py --offscreen
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
from robot_descriptions import h1_mj_description

from humanoid_rl.envs.mujoco.terrain import inject_terrain


# ─── Joint layout (19 actuators) ─────────────────────────────────────────────
#  0: left_hip_yaw      5: right_hip_yaw     10: torso
#  1: left_hip_roll     6: right_hip_roll     11-14: left arm
#  2: left_hip_pitch    7: right_hip_pitch    15-18: right arm
#  3: left_knee         8: right_knee
#  4: left_ankle        9: right_ankle

# PD gains
KP = np.array([
    200, 200, 200, 300, 40,   # left leg
    200, 200, 200, 300, 40,   # right leg
    200,                       # torso
    40, 40, 18, 18,            # left arm
    40, 40, 18, 18,            # right arm
], dtype=np.float64)

KD = np.array([
    5, 5, 8, 10, 2,
    5, 5, 8, 10, 2,
    5,
    2, 2, 1, 1,
    2, 2, 1, 1,
], dtype=np.float64)

# Default standing pose
Q_DEFAULT = np.zeros(19, dtype=np.float64)
Q_DEFAULT[3]  = 0.05    # left knee (very slight bend)
Q_DEFAULT[8]  = 0.05    # right knee
Q_DEFAULT[12] = 0.2     # left shoulder roll
Q_DEFAULT[16] = -0.2    # right shoulder roll
Q_DEFAULT[14] = 0.3     # left elbow
Q_DEFAULT[18] = 0.3     # right elbow

ROOT_Z = 1.06


# ─── Keyboard ────────────────────────────────────────────────────────────────

class KeyState:
    def __init__(self):
        self.fwd = self.bwd = self.left = self.right = False
        self.reset = self.quit = False

    @property
    def vx(self):
        return (1.0 if self.fwd else 0.0) + (-0.5 if self.bwd else 0.0)

    @property
    def yaw(self):
        return (0.5 if self.left else 0.0) + (-0.5 if self.right else 0.0)


def make_key_cb(ks):
    def cb(keycode):
        if keycode in (87, 265):   ks.fwd = not ks.fwd
        elif keycode in (83, 267): ks.bwd = not ks.bwd
        elif keycode in (65, 263): ks.left = not ks.left
        elif keycode in (68, 262): ks.right = not ks.right
        elif keycode == 32:        ks.reset = True
        elif keycode in (81, 256): ks.quit = True
    return cb


# ─── Controller ──────────────────────────────────────────────────────────────

class H1Controller:
    """Balance + walking controller for H1."""

    def __init__(self, model):
        self.model = model
        self.nu = model.nu

        # Body IDs
        self.pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        self.lankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_link")
        self.rankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_link")

        self.total_mass = sum(model.body_mass[i] for i in range(1, model.nbody))
        self.g = abs(model.opt.gravity[2])

        # Walking state
        self.phase = 0.0
        self.freq = 1.8

    def step(self, data, cmd_vx, cmd_yaw, dt):
        """Compute PD torques + balance forces and apply them."""

        # ── 1. Walking target ────────────────────────────────────
        q_target = Q_DEFAULT.copy()

        if abs(cmd_vx) > 0.05 or abs(cmd_yaw) > 0.05:
            self.phase += dt * self.freq * 2 * np.pi
            amp = np.clip(abs(cmd_vx) * 0.35, 0.0, 0.45)
            sgn = 1.0 if cmd_vx >= 0 else -1.0

            sl = np.sin(self.phase)
            sr = np.sin(self.phase + np.pi)

            q_target[2] += sgn * amp * sl * 0.5    # L hip pitch
            q_target[7] += sgn * amp * sr * 0.5    # R hip pitch
            q_target[3] += max(0, sl) * amp * 0.35  # L knee
            q_target[8] += max(0, sr) * amp * 0.35  # R knee
            q_target[4] += -sl * amp * 0.15          # L ankle
            q_target[9] += -sr * amp * 0.15          # R ankle

            if abs(cmd_yaw) > 0.05:
                q_target[0] += cmd_yaw * 0.12
                q_target[5] -= cmd_yaw * 0.12

            q_target[11] += -sgn * amp * sl * 0.15
            q_target[15] += -sgn * amp * sr * 0.15

        # ── 2. PD joint torques ──────────────────────────────────
        q = data.qpos[7:]
        dq = data.qvel[6:]
        torques = KP * (q_target - q) - KD * dq

        for i in range(self.nu):
            lo, hi = self.model.actuator_ctrlrange[i]
            torques[i] = np.clip(torques[i], lo, hi)
        data.ctrl[:] = torques

        # ── 3. Balance: virtual force on pelvis ──────────────────
        self._apply_balance_force(data)

    def _apply_balance_force(self, data):
        """Apply virtual stabilizing force at pelvis to keep CoM over feet."""
        # Compute CoM
        com = np.zeros(3)
        com_vel = np.zeros(3)
        tm = 0.0
        for i in range(1, self.model.nbody):
            m = self.model.body_mass[i]
            com += data.xipos[i] * m
            v6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model, data, mujoco.mjtObj.mjOBJ_BODY, i, v6, 0)
            com_vel += v6[3:6] * m
            tm += m
        com /= tm
        com_vel /= tm

        # Support center
        sup = (data.xpos[self.lankle_id] + data.xpos[self.rankle_id]) / 2.0

        # CoM error (horizontal only)
        dx = com[0] - sup[0]
        dy = com[1] - sup[1]
        dvx, dvy = com_vel[0], com_vel[1]

        # Root orientation errors
        qw, qx, qy, qz = data.qpos[3:7]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))
        roll_rate  = data.qvel[3]
        pitch_rate = data.qvel[4]

        # Horizontal stabilizing forces (PD on CoM position)
        # Robot weight ~504N, CoM at 1m → need strong forces
        kp_h, kd_h = 2000.0, 500.0
        fx = np.clip(-(kp_h * dx + kd_h * dvx), -500, 500)
        fy = np.clip(-(kp_h * dy + kd_h * dvy), -500, 500)

        # Angular stabilizing torques (PD on orientation)
        kp_a, kd_a = 500.0, 100.0
        tx = np.clip(-(kp_a * roll  + kd_a * roll_rate),  -200, 200)
        ty = np.clip(-(kp_a * pitch + kd_a * pitch_rate), -200, 200)

        # Apply to pelvis: [torque_x, torque_y, torque_z, force_x, force_y, force_z]
        data.xfrc_applied[self.pelvis_id, :] = [tx, ty, 0, fx, fy, 0]


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_h1():
    xml_path = h1_mj_description.MJCF_PATH
    xml_dir = str(Path(xml_path).parent)

    with open(xml_path) as f:
        xml = f.read()

    xml = xml.replace('meshdir="assets"', f'meshdir="{xml_dir}/assets"')
    xml = xml.replace('texturedir="assets"', f'texturedir="{xml_dir}/assets"')

    import re
    fb = '  <visual>\n    <global offwidth="1280" offheight="720"/>\n  </visual>'
    if not re.search(r'^\s*<visual\s*>', xml, re.MULTILINE):
        xml = re.sub(r'(<mujoco[^>]*>)', r'\1\n' + fb, xml, count=1)

    xml = inject_terrain(xml, "flat")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data


def reset(model, data, ctrl):
    mujoco.mj_resetData(model, data)
    data.qpos[2] = ROOT_Z
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[7:] = Q_DEFAULT
    data.qvel[:] = 0
    data.xfrc_applied[:] = 0
    ctrl.phase = 0.0
    mujoco.mj_forward(model, data)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="H1 Interactive Controller")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--output", default="videos/h1_demo.mp4")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    model, data = load_h1()
    ctrl = H1Controller(model)
    frame_skip = 10
    ctrl_dt = model.opt.timestep * frame_skip

    if args.offscreen:
        # ── Offscreen recording ──────────────────────────────
        import imageio
        renderer = mujoco.Renderer(model, 540, 960)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth, cam.elevation, cam.distance = 135, -20, 4.5
        cam.lookat[:] = [0, 0, 0.9]
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        reset(model, data, ctrl)
        frames = []
        n = args.steps

        # Phase 1: Stand (2s)
        n1 = min(100, n)
        print(f"[1/3] Standing {n1} steps...")
        for step in range(n1):
            ctrl.step(data, 0.0, 0.0, ctrl_dt)
            for _ in range(frame_skip): mujoco.mj_step(model, data)
            cam.lookat[0] = float(data.qpos[0])
            renderer.update_scene(data, camera=cam, scene_option=opt)
            frames.append(renderer.render().copy())
        tz = data.xpos[ctrl.torso_id, 2]
        print(f"  torso_z={tz:.3f}")

        # Phase 2: Walk (6s)
        n2 = min(300, n - n1)
        if n2 > 0:
            print(f"[2/3] Walking {n2} steps...")
            for step in range(n2):
                ctrl.step(data, 0.8, 0.0, ctrl_dt)
                for _ in range(frame_skip): mujoco.mj_step(model, data)
                cam.lookat[0] = float(data.qpos[0])
                renderer.update_scene(data, camera=cam, scene_option=opt)
                frames.append(renderer.render().copy())
            tz = data.xpos[ctrl.torso_id, 2]
            vx = data.qvel[0]
            print(f"  torso_z={tz:.3f} vx={vx:+.3f}")

        # Phase 3: Stand again
        n3 = n - n1 - max(n2, 0)
        if n3 > 0:
            print(f"[3/3] Standing {n3} steps...")
            for step in range(n3):
                ctrl.step(data, 0.0, 0.0, ctrl_dt)
                for _ in range(frame_skip): mujoco.mj_step(model, data)
                cam.lookat[0] = float(data.qpos[0])
                renderer.update_scene(data, camera=cam, scene_option=opt)
                frames.append(renderer.render().copy())
            tz = data.xpos[ctrl.torso_id, 2]
            print(f"  torso_z={tz:.3f}")

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(out), frames, fps=50, quality=8, macro_block_size=1)
        mb = out.stat().st_size / 1024 / 1024
        print(f"\n✅ Saved: {out} ({len(frames)} frames, {mb:.1f} MB)")
        renderer.close()
        return

    # ── Interactive viewer ───────────────────────────────────
    reset(model, data, ctrl)
    ks = KeyState()

    print("=" * 50)
    print("  Unitree H1 — Interactive Controller")
    print("=" * 50)
    print("  W = Forward    S = Backward")
    print("  A = Turn Left  D = Turn Right")
    print("  SPACE = Reset  Q = Quit")
    print("=" * 50)

    with mujoco.viewer.launch_passive(model, data, key_callback=make_key_cb(ks)) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
        viewer.cam.distance = 4.5
        viewer.cam.lookat[:] = [0, 0, 0.9]

        step = 0
        t_prev = time.time()

        while viewer.is_running() and not ks.quit:
            if ks.reset:
                reset(model, data, ctrl)
                ks.reset = False
                step = 0
                print("[Reset]")

            ctrl.step(data, ks.vx, ks.yaw, ctrl_dt)
            for _ in range(frame_skip):
                mujoco.mj_step(model, data)
            step += 1

            viewer.cam.lookat[0] = float(data.qpos[0])
            viewer.cam.lookat[1] = float(data.qpos[1])

            if step % 200 == 0:
                tz = data.xpos[ctrl.torso_id, 2]
                vx = data.qvel[0]
                print(f"  step={step:5d} torso_z={tz:.3f} vx={vx:+.3f} cmd=[{ks.vx:.1f},{ks.yaw:.1f}]")

            # Auto-reset if fallen
            tz = data.xpos[ctrl.torso_id, 2]
            if tz < 0.4:
                print(f"  [Fell at step {step}] Auto-reset")
                reset(model, data, ctrl)
                step = 0

            viewer.sync()

            t_now = time.time()
            if t_now - t_prev < ctrl_dt:
                time.sleep(ctrl_dt - (t_now - t_prev))
            t_prev = time.time()

    print("\nDone.")


if __name__ == "__main__":
    main()
