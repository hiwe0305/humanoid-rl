#!/usr/bin/env python3
"""
Play checkpoint inference for H1 humanoid with real-time MuJoCo simulation.

Visualization modes:
    1. WebGL 3D Viewer (CHÍNH THỨC CỦA MUJOCO - KHUYÊN DÙNG NHẤT):
       Sử dụng WebGL & WebAssembly (WASM) của MuJoCo Studio, cho phép xoay chuột 3D 360 độ:
       PYTHONPATH=. venv/bin/python scripts/play_checkpoint.py \\
           --checkpoint runs/h1_stand_v4/checkpoints/checkpoint_latest.pt \\
           --webgl --port 8080

    2. MJPEG Web Streamer:
       Server render bằng EGL và stream hình ảnh liên tục qua trình duyệt:
       PYTHONPATH=. MUJOCO_GL=egl venv/bin/python scripts/play_checkpoint.py \\
           --checkpoint runs/h1_stand_v4/checkpoints/checkpoint_latest.pt \\
           --web --port 8080

    3. Record MP4 Video:
       Xuất video MP4 để tải về xem lại trong VS Code Explorer:
       PYTHONPATH=. MUJOCO_GL=egl venv/bin/python scripts/play_checkpoint.py \\
           --checkpoint runs/h1_stand_v4/checkpoints/checkpoint_latest.pt \\
           --record-video runs/h1_stand_v4/stand_demo.mp4 --steps 500

    4. Desktop GLFW Viewer (Chỉ dùng khi máy có màn hình hiển thị Desktop X11/Wayland thật):
       PYTHONPATH=. MUJOCO_GL=glfw venv/bin/python scripts/play_checkpoint.py \\
           --checkpoint runs/h1_stand_v4/checkpoints/checkpoint_latest.pt \\
           --interactive
"""
import sys
import time
import argparse
from pathlib import Path
import enum

# Backport StrEnum for Python 3.10 compatibility with MuJoCo Studio Web Server
if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        pass
    enum.StrEnum = StrEnum

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch
import numpy as np
import mujoco

try:
    import mujoco.viewer
except Exception:
    pass


def setup_camera(camera: mujoco.MjvCamera, track_body_id: int, distance: float = 2.8, elevation: float = -12.0, azimuth: float = 90.0):
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = track_body_id
    camera.distance = distance
    camera.elevation = elevation
    camera.azimuth = azimuth


def run_webgl_viewer(env, ac, device, port: int = 8080, speed_factor: float = 1.0, max_steps: int = 10000):
    """Launch official MuJoCo Studio WebGL / WASM interactive 3D viewer."""
    from mujoco.experimental.studio import launch_passive, viewer_protocol, messages, viewer_app

    config = viewer_protocol.ViewerConfig(gfx="web", http_port=port)
    handle = launch_passive.launch_passive(config, viewer_plugins=[viewer_app.ViewerApp()])
    handle.send_to_viewer(messages.ModelEvent(model=env.model))

    print("\n" + "="*68)
    print(f"  MUJOCO OFFICIAL WEBGL (WASM) 3D VIEWER IS RUNNING!")
    print(f"  --> URL:  http://localhost:{port}  (Hoặc http://<IP-Server>:{port})")
    print(f"  - Nếu bạn dùng VS Code: Click 'Open in Browser' khi có thông báo.")
    print(f"  - Render 3D thời gian thực trực tiếp trên GPU máy của bạn qua WebGL.")
    print(f"  - Chuột trái: Xoay góc nhìn 3D | Chuột phải: Zoom | Shift+Chuột: Pan.")
    print(f"  - Bấm Ctrl+C trong terminal bất kỳ lúc nào để dừng.")
    print("="*68 + "\n")

    obs, _ = env.reset()
    dt = env.dt
    step_count = 0
    episodes = 0
    ep_return = 0.0
    ep_len = 0

    try:
        while handle.is_running() and step_count < max_steps:
            step_start = time.time()

            # Policy deterministic action
            action, _, _ = ac.act(obs, device=device)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            step_count += 1

            # Sync physics state to browser WebGL client
            handle.sync(env.model, env.data)

            if step_count % 50 == 0:
                torso_z = info.get("torso_height", 0.0)
                vx = info.get("x_velocity", 0.0)
                print(f"  [WebGL 3D] Step {step_count:5d} | Torso Z: {torso_z:.3f}m | Vx: {vx:+.2f}m/s | Return: {ep_return:.1f}")

            if terminated or truncated:
                episodes += 1
                status = "COMPLETED" if truncated else "FALL"
                print(f"  --> Episode {episodes} ended [{status}]: {ep_len} steps, Return={ep_return:.1f}")
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0

            # Real-time synchronization (50 Hz)
            target_sleep = (dt / max(speed_factor, 0.01)) - (time.time() - step_start)
            if target_sleep > 0:
                time.sleep(target_sleep)
    except KeyboardInterrupt:
        print("\n[+] Stopping WebGL viewer...")
    finally:
        handle.close()


def run_web_streamer(env, ac, device, port: int = 8080, speed_factor: float = 1.0, max_steps: int = 10000, width: int = 640, height: int = 480):
    """Launch MJPEG HTTP video streamer using server-side EGL rendering."""
    import http.server
    import socketserver
    import threading
    import cv2

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    setup_camera(camera, env._pelvis_bid, distance=2.6, elevation=-10.0, azimuth=90.0)

    state = {
        "frame_bytes": None,
        "lock": threading.Lock(),
        "running": True,
        "stats": "Initializing...",
    }

    class StreamingHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while state["running"]:
                        with state["lock"]:
                            frame_data = state["frame_bytes"]
                        if frame_data is not None:
                            self.wfile.write(b"--FRAME\r\n")
                            self.send_header("Content-Type", "image/jpeg")
                            self.send_header("Content-Length", str(len(frame_data)))
                            self.end_headers()
                            self.wfile.write(frame_data + b"\r\n")
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            elif self.path == "/stats":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with state["lock"]:
                    stats_json = f'{{"info": "{state["stats"]}"}}'.encode("utf-8")
                self.wfile.write(stats_json)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>H1 Humanoid RL - Live Simulation Stream</title>
    <style>
        * { box-sizing: border-box; }
        body { background: #090d16; color: #f8fafc; font-family: ui-sans-serif, system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; margin: 0; padding: 24px; }
        h1 { margin: 0 0 6px 0; font-size: 26px; color: #38bdf8; font-weight: 700; }
        .subtitle { color: #94a3b8; margin-bottom: 24px; font-size: 14px; }
        .viewer-card { background: #131b2e; border-radius: 16px; padding: 18px; box-shadow: 0 20px 40px rgba(0,0,0,0.6); border: 1px solid #1e293b; max-width: 680px; width: 100%; text-align: center; }
        .stream-img { display: block; border-radius: 10px; width: 100%; max-width: 640px; height: auto; margin: 0 auto; background: #000; border: 1px solid #334155; }
        .telemetry { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; justify-content: center; font-size: 13px; }
        .badge { background: #0a0f1d; padding: 8px 14px; border-radius: 8px; border: 1px solid #1e293b; color: #cbd5e1; }
        .badge b { color: #38bdf8; font-family: monospace; font-size: 14px; }
        .indicator { display: inline-block; width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-right: 6px; box-shadow: 0 0 8px #22c55e; }
    </style>
</head>
<body>
    <h1>Unitree H1 Humanoid Simulation</h1>
    <div class="subtitle"><span class="indicator"></span>Live Simulation Stream (MuJoCo EGL @ 50 FPS)</div>
    <div class="viewer-card">
        <img class="stream-img" src="/stream.mjpg" alt="Live Simulation Stream">
        <div class="telemetry">
            <div class="badge">Telemetry: <b id="stat-text">Connecting...</b></div>
        </div>
    </div>
    <script>
        setInterval(async () => {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                document.getElementById('stat-text').innerText = data.info;
            } catch(e) {}
        }, 500);
    </script>
</body>
</html>"""
                self.wfile.write(html.encode("utf-8"))

    class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    try:
        server = ThreadedHTTPServer(("0.0.0.0", port), StreamingHandler)
    except OSError:
        print(f"[-] ERROR: Port {port} is already in use. Please specify another port: --port {port+1}")
        return

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n" + "="*68)
    print(f"  LIVE WEB MJPEG STREAMING IS RUNNING!")
    print(f"  --> URL:  http://localhost:{port}  (Hoặc http://<IP-Server>:{port})")
    print("="*68 + "\n")

    obs, _ = env.reset()
    dt = env.dt
    step_count = 0
    episodes = 0
    ep_return = 0.0
    ep_len = 0

    try:
        while state["running"] and step_count < max_steps:
            step_start = time.time()

            action, _, _ = ac.act(obs, device=device)

            renderer.update_scene(env.data, camera=camera)
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            torso_z = float(env._get_body_z(env._torso_body))
            vx = float(env.data.qvel[0])

            cv2.putText(
                frame_bgr,
                f"Step: {step_count} | Z: {torso_z:.2f}m | Vx: {vx:+.2f}m/s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if torso_z >= 0.85 else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            _, jpeg_buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            with state["lock"]:
                state["frame_bytes"] = jpeg_buf.tobytes()
                state["stats"] = f"Ep {episodes+1} | Step {step_count} | Z={torso_z:.2f}m | Vx={vx:+.2f}m/s | Ret={ep_return:.1f}"

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            step_count += 1

            if step_count % 50 == 0:
                print(f"  [Live Web] Step {step_count:5d} | Torso Z: {torso_z:.3f}m | Vx: {vx:+.2f}m/s | Ep Ret: {ep_return:.1f}")

            if terminated or truncated:
                episodes += 1
                status = "COMPLETED" if truncated else "FALL"
                print(f"  --> Episode {episodes} ended [{status}]: {ep_len} steps, Return={ep_return:.1f}")
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0

            target_sleep = (dt / max(speed_factor, 0.01)) - (time.time() - step_start)
            if target_sleep > 0:
                time.sleep(target_sleep)
    except KeyboardInterrupt:
        print("\n[+] Stopping web server...")
    finally:
        state["running"] = False
        server.shutdown()
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Play H1 checkpoint inference")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--webgl", action="store_true", help="Launch official MuJoCo WebGL 3D interactive viewer (WASM/Filament)")
    parser.add_argument("--cockpit", "--teleop", action="store_true", help="Launch Cyberpunk Teleoperation Cockpit with W/A/S/D directional control")
    parser.add_argument("--web", action="store_true", help="Launch lightweight MJPEG web simulation streamer")
    parser.add_argument("--port", type=int, default=8080, help="Port for web/webgl viewer (default: 8080)")
    parser.add_argument("--interactive", action="store_true", help="Launch desktop 3D MuJoCo GLFW viewer (requires active desktop display)")
    parser.add_argument("--record-video", type=str, default=None, help="Save simulation to MP4 video file (e.g. demo.mp4)")
    parser.add_argument("--steps", type=int, default=10000, help="Number of steps to run (default: 10000)")
    parser.add_argument("--task", type=str, default="auto", choices=["auto", "stand", "walk"], help="Task mode (default: auto)")
    parser.add_argument("--target-speed", type=float, default=1.0, help="Target walking speed (m/s) if walk task")
    parser.add_argument("--speed-factor", type=float, default=1.0, help="Real-time playback speed factor (e.g. 1.0 = real-time, 0.5 = 2x slow-mo)")
    parser.add_argument("--device", default=None, help="Device (cpu/cuda)")
    parser.add_argument("--width", type=int, default=640, help="Video width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Video height (default: 480)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)

    if not ckpt_path.exists():
        print(f"[-] ERROR: Checkpoint not found: {ckpt_path}")
        return

    if args.cockpit or args.web or args.webgl:
        import scripts.play_teleop as play_teleop
        sys.argv = [
            "play_teleop.py",
            "--walk-ckpt", str(ckpt_path),
            "--port", str(args.port),
            "--device", device,
        ]
        play_teleop.main()
        return

    # Load checkpoint
    print(f"\n[+] Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Extract state dict
    ac_state = checkpoint.get("ac_state", checkpoint)
    global_step = checkpoint.get("global_step", 0)
    print(f"[+] Checkpoint trained to global_step: {global_step:,}")

    # Inspect input observation dimension
    if "actor.net.0.weight" in ac_state:
        obs_dim = ac_state["actor.net.0.weight"].shape[1]
    else:
        obs_dim = 246

    print(f"[+] Detected policy obs_dim: {obs_dim}")

    # Determine task
    if args.task == "auto":
        task_name = "walk" if obs_dim >= 252 else "stand"
    else:
        task_name = args.task

    print(f"[+] Task mode: {task_name.upper()} (target_speed={args.target_speed if task_name == 'walk' else 0.0} m/s)")

    # Build ActorCritic
    from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic
    act_dim = 19
    ac = ActorCritic(obs_dim, act_dim)
    ac.load_state_dict(ac_state)
    ac.to(device)
    ac.eval()
    print(f"[+] Policy loaded on {device}")

    # Build Environment
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    if task_name == "walk":
        from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
        task_cfg = WalkTask(target_speed=args.target_speed, max_steps=args.steps).to_cfg()
    else:
        from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
        task_cfg = StandTask(max_steps=args.steps).to_cfg()

    env = H1Env(
        task_cfg=task_cfg,
        domain_rand_cfg=None,
        push_rand_cfg=None,
        render_mode="rgb_array" if (args.record_video or args.web) else None,
        frame_skip=10,
        history_len=3,
    )

    obs, _ = env.reset()
    dt = env.dt  # 0.02s = 50 Hz

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 1: Official MuJoCo Studio WebGL / WASM 3D Viewer
    # ──────────────────────────────────────────────────────────────────────────
    if args.webgl:
        run_webgl_viewer(
            env=env,
            ac=ac,
            device=device,
            port=args.port,
            speed_factor=args.speed_factor,
            max_steps=args.steps,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 2: MJPEG Web Streamer
    # ──────────────────────────────────────────────────────────────────────────
    elif args.web:
        run_web_streamer(
            env=env,
            ac=ac,
            device=device,
            port=args.port,
            speed_factor=args.speed_factor,
            max_steps=args.steps,
            width=args.width,
            height=args.height,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 3: Interactive Desktop 3D GUI (GLFW)
    # ──────────────────────────────────────────────────────────────────────────
    elif args.interactive:
        print("\n" + "="*60)
        print("  INTERACTIVE 3D VIEWER (Real-Time 1.0x Simulation)")
        print("  Physics rate: 50 Hz | Time step dt = 0.02s")
        print("  Press Space to pause/resume, Double-click body to track")
        print("  Press ESC or close window to exit")
        print("="*60 + "\n")

        import os
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            print("[-] CẢNH BÁO: Không tìm thấy biến môi trường DISPLAY hoặc WAYLAND_DISPLAY.")
            print("    Chế độ --interactive cần màn hình đồ hoạ (X11 / Wayland) để mở cửa sổ 3D GLFW.")
            print("    👉 Để xem 3D trực tiếp trong trình duyệt qua WebGL (không cần desktop DISPLAY):")
            print("       PYTHONPATH=. venv/bin/python scripts/play_checkpoint.py \\")
            print(f"           --checkpoint {args.checkpoint} --webgl\n")
            print("    👉 Hoặc xuất video MP4:")
            print(f"       PYTHONPATH=. MUJOCO_GL=egl venv/bin/python scripts/play_checkpoint.py \\")
            print(f"           --checkpoint {args.checkpoint} --record-video stand_demo.mp4 --steps 500\n")
            env.close()
            return

        try:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        except Exception as e:
            print(f"[-] ERROR: Failed to open GLFW window: {e}")
            env.close()
            return

        with viewer:
            setup_camera(viewer.cam, env._pelvis_bid, distance=2.8, elevation=-10.0, azimuth=90.0)

            step_count = 0
            episodes = 0
            ep_return = 0.0
            ep_len = 0

            while viewer.is_running() and step_count < args.steps:
                step_start = time.time()

                action, _, _ = ac.act(obs, device=device)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                ep_len += 1
                step_count += 1

                viewer.sync()

                if step_count % 50 == 0:
                    torso_z = info.get("torso_height", 0.0)
                    vx = info.get("x_velocity", 0.0)
                    contacts = info.get("foot_contacts", [0, 0])
                    print(f"  Step {step_count:4d}/{args.steps} | Torso Z: {torso_z:.3f}m | Vx: {vx:+.2f}m/s | Feet: {contacts} | Ep Ret: {ep_return:.1f}")

                if terminated or truncated:
                    episodes += 1
                    status = "Truncated (Completed full episode!)" if truncated else "Terminated (Fell down)"
                    print(f"\n[Episode {episodes}] {status} | Length: {ep_len} steps | Total Return: {ep_return:.1f}")
                    obs, _ = env.reset()
                    ep_return = 0.0
                    ep_len = 0

                target_sleep = (dt / max(args.speed_factor, 0.01)) - (time.time() - step_start)
                if target_sleep > 0:
                    time.sleep(target_sleep)

        print("\n[+] Viewer closed.")

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 4: Video Recording (Offscreen EGL / Headless friendly)
    # ──────────────────────────────────────────────────────────────────────────
    elif args.record_video:
        import cv2
        video_path = Path(args.record_video)
        video_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print(f"  RECORDING SIMULATION VIDEO TO: {video_path}")
        print(f"  Resolution: {args.width}x{args.height} | FPS: 50 | Total Steps: {args.steps}")
        print("="*60 + "\n")

        renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
        camera = mujoco.MjvCamera()
        setup_camera(camera, env._pelvis_bid, distance=2.6, elevation=-10.0, azimuth=90.0)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 50.0 * args.speed_factor
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (args.width, args.height))

        step_count = 0
        episodes = 0
        ep_return = 0.0
        ep_len = 0

        while step_count < args.steps:
            action, _, _ = ac.act(obs, device=device)

            renderer.update_scene(env.data, camera=camera)
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            torso_z = float(env._get_body_z(env._torso_body))
            vx = float(env.data.qvel[0])
            cv2.putText(
                frame_bgr,
                f"Step: {step_count} | Z: {torso_z:.2f}m | Vx: {vx:+.2f}m/s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if torso_z >= 0.85 else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame_bgr)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_len += 1
            step_count += 1

            if step_count % 50 == 0:
                print(f"  Rendering step {step_count}/{args.steps} ({(step_count/args.steps)*100:.0f}%)...")

            if terminated or truncated:
                episodes += 1
                status = "COMPLETED" if truncated else "FALL"
                print(f"  --> Episode {episodes} ended [{status}]: {ep_len} steps, Return={ep_return:.1f}")
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0

        writer.release()
        print(f"\n[+] Video saved successfully: {video_path} ({video_path.stat().st_size / (1024*1024):.2f} MB)")

    # ──────────────────────────────────────────────────────────────────────────
    # MODE 5: Headless Fast Evaluation
    # ──────────────────────────────────────────────────────────────────────────
    else:
        print(f"\n[+] Running headless inference for {args.steps} steps...")
        step_count = 0
        episodes = 0
        ep_returns = []
        ep_lens = []
        cur_return = 0.0
        cur_len = 0

        while step_count < args.steps:
            action, _, _ = ac.act(obs, device=device)
            obs, reward, terminated, truncated, info = env.step(action)
            cur_return += reward
            cur_len += 1
            step_count += 1

            if terminated or truncated:
                episodes += 1
                ep_returns.append(cur_return)
                ep_lens.append(cur_len)
                status = "Truncated (Max Steps)" if truncated else "Terminated (Fell)"
                print(f"  Episode {episodes}: return={cur_return:.1f}, steps={cur_len}, status={status}, torso_z={info.get('torso_height', 0):.3f}m")
                cur_return = 0.0
                cur_len = 0
                obs, _ = env.reset()

        print(f"\n{'='*50}")
        print(f"  Inference Benchmark Complete!")
        print(f"  Total episodes: {episodes}")
        if ep_returns:
            print(f"  Avg return:     {np.mean(ep_returns):.1f}")
            print(f"  Avg length:     {np.mean(ep_lens):.1f} steps")
            print(f"  Max return:     {np.max(ep_returns):.1f}")
        print(f"{'='*50}\n")

    env.close()


if __name__ == "__main__":
    main()
