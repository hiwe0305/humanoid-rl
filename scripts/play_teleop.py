#!/usr/bin/env python3
"""
scripts/play_teleop.py
Interactive Cyberpunk Teleoperation Cockpit for Unitree H1 Humanoid.

Features:
  - Real-time 50 Hz bipedal simulation with dynamic camera tracking
  - Full directional control: Tiến (W), Lùi (S), Rẽ trái (A), Rẽ phải (D), Đứng thăng bằng (Space)
  - Real-time live HUD: Speedometer, Torso Height, Foot Contacts, Artificial Horizon, Yaw Compass
  - Multiple camera modes: Chase Cam (sau lưng), Orbit Cam, Side View, Front View
  - Dual policy blending: Seamlessly switch between Walk Policy and Stand Balance Policy

Usage:
    PYTHONPATH=. MUJOCO_GL=egl venv/bin/python scripts/play_teleop.py --port 8080
"""
import sys
import time
import json
import argparse
import threading
import http.server
import socketserver
from pathlib import Path
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, str(Path(__file__).parents[1]))

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import numpy as np
import mujoco
import cv2

from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic


# ─── HTML & CSS Cockpit UI ──────────────────────────────────────────────────
COCKPIT_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unitree H1 - Cyberpunk Teleoperation Cockpit</title>
    <style>
        :root {
            --bg-base: #060911;
            --bg-card: rgba(13, 20, 36, 0.75);
            --border-glow: rgba(0, 242, 254, 0.25);
            --cyan-glow: #00f2fe;
            --emerald-glow: #10b981;
            --amber-glow: #f59e0b;
            --rose-glow: #f43f5e;
            --text-main: #f8fafc;
            --text-dim: #94a3b8;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; user-select: none; }
        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", monospace, sans-serif;
            min-height: 100vh;
            overflow-x: hidden;
            display: flex;
            flex-direction: column;
            background-image: 
                radial-gradient(circle at 50% 0%, rgba(0, 242, 254, 0.08) 0%, transparent 60%),
                linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 100% 100%, 40px 40px, 40px 40px;
        }

        /* Top Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 28px;
            background: rgba(10, 15, 28, 0.85);
            border-bottom: 1px solid var(--border-glow);
            backdrop-filter: blur(12px);
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .brand-logo {
            font-size: 20px;
            font-weight: 900;
            letter-spacing: 1.5px;
            background: linear-gradient(135deg, #00f2fe, #4facfe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .status-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 999px;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: var(--emerald-glow);
        }
        .pulse-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--emerald-glow);
            box-shadow: 0 0 10px var(--emerald-glow);
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(0.9); opacity: 0.8; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.9); opacity: 0.8; }
        }

        /* Main Workspace Grid */
        .cockpit-grid {
            display: grid;
            grid-template-columns: 280px 1fr 340px;
            gap: 20px;
            padding: 20px 28px;
            flex: 1;
            align-items: start;
        }
        @media (max-width: 1200px) {
            .cockpit-grid { grid-template-columns: 1fr; }
        }

        /* Card panels */
        .glass-card {
            background: var(--bg-card);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            padding: 18px;
            backdrop-filter: blur(16px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }
        .card-header {
            font-size: 13px;
            font-weight: 700;
            letter-spacing: 1px;
            color: var(--cyan-glow);
            text-transform: uppercase;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Viewport Center Screen */
        .viewport-wrapper {
            position: relative;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--cyan-glow);
            box-shadow: 0 0 35px rgba(0, 242, 254, 0.2);
            background: #000;
        }
        .video-stream {
            width: 100%;
            height: auto;
            display: block;
            min-height: 480px;
            object-fit: cover;
        }

        /* Viewport HUD Overlays */
        .hud-top-bar {
            position: absolute;
            top: 14px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 16px;
            background: rgba(6, 9, 17, 0.7);
            border: 1px solid var(--border-glow);
            padding: 6px 16px;
            border-radius: 20px;
            backdrop-filter: blur(8px);
            font-size: 13px;
            font-family: monospace;
        }
        .hud-compass {
            color: var(--cyan-glow);
            font-weight: bold;
        }

        /* Camera switcher pills */
        .cam-bar {
            position: absolute;
            bottom: 14px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 8px;
            background: rgba(6, 9, 17, 0.8);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .cam-btn {
            background: transparent;
            border: 1px solid transparent;
            color: var(--text-dim);
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .cam-btn:hover { color: #fff; background: rgba(255, 255, 255, 0.05); }
        .cam-btn.active {
            background: rgba(0, 242, 254, 0.2);
            border-color: var(--cyan-glow);
            color: var(--cyan-glow);
        }

        /* Telemetry Gauges */
        .gauge-metric {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 12px;
        }
        .metric-title {
            font-size: 11px;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 800;
            font-family: monospace;
            color: #fff;
        }
        .metric-unit {
            font-size: 13px;
            color: var(--text-dim);
            margin-left: 4px;
            font-weight: 500;
        }

        /* Foot contact indicators */
        .foot-indicators {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 14px;
        }
        .foot-box {
            padding: 12px 8px;
            border-radius: 10px;
            text-align: center;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid rgba(255, 255, 255, 0.08);
            background: rgba(0, 0, 0, 0.3);
            transition: all 0.15s;
        }
        .foot-box.grounded {
            background: rgba(16, 185, 129, 0.25);
            border-color: var(--emerald-glow);
            color: var(--emerald-glow);
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
        }
        .foot-box.air {
            background: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.3);
            color: var(--amber-glow);
        }

        /* Directional Control Pad (D-PAD) */
        .dpad-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            margin: 20px 0;
        }
        .dpad-row {
            display: flex;
            gap: 8px;
        }
        .ctrl-btn {
            width: 72px;
            height: 60px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-radius: 12px;
            color: #fff;
            font-size: 20px;
            font-weight: 900;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 2px;
            transition: all 0.15s;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        }
        .ctrl-btn span {
            font-size: 10px;
            font-weight: 600;
            color: var(--text-dim);
            letter-spacing: 0.5px;
        }
        .ctrl-btn:hover {
            background: rgba(0, 242, 254, 0.15);
            border-color: var(--cyan-glow);
            transform: translateY(-2px);
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.4);
        }
        .ctrl-btn:active, .ctrl-btn.active {
            background: var(--cyan-glow);
            color: #000;
            border-color: var(--cyan-glow);
            transform: translateY(1px);
            box-shadow: 0 0 25px var(--cyan-glow);
        }
        .ctrl-btn:active span, .ctrl-btn.active span {
            color: #000;
        }
        .ctrl-btn.btn-stop {
            width: 152px;
            border-color: rgba(244, 63, 94, 0.4);
            background: rgba(244, 63, 94, 0.1);
        }
        .ctrl-btn.btn-stop:hover {
            border-color: var(--rose-glow);
            background: rgba(244, 63, 94, 0.25);
            box-shadow: 0 0 15px rgba(244, 63, 94, 0.4);
        }
        .ctrl-btn.btn-stop.active {
            background: var(--rose-glow);
            color: #fff;
        }

        /* Sliders */
        .slider-group {
            margin-top: 14px;
        }
        .slider-label {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }
        input[type="range"] {
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
            accent-color: var(--cyan-glow);
        }

        /* Quick shortcuts guide */
        .shortcut-tip {
            margin-top: 16px;
            padding: 10px;
            background: rgba(0, 0, 0, 0.25);
            border-radius: 8px;
            font-size: 11px;
            color: var(--text-dim);
            line-height: 1.6;
        }
        .kbd {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            color: var(--cyan-glow);
            font-weight: bold;
            font-family: monospace;
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <span class="brand-logo">H1 CYBER-TELEOP</span>
            <div class="status-badge"><span class="pulse-dot"></span> LIVE 50 Hz</div>
        </div>
        <div style="font-size: 12px; color: var(--text-dim);">
            Task: <b style="color: #fff;" id="task-badge">BIPEDAL LOCOMOTION</b>
        </div>
    </header>

    <div class="cockpit-grid">
        <!-- Left Panel: Telemetry HUD -->
        <div class="glass-card">
            <div class="card-header">Telemetry Gauges</div>

            <div class="gauge-metric">
                <div class="metric-title">Vận tốc tiến (Vx)</div>
                <div><span class="metric-value" id="val-vx">0.00</span><span class="metric-unit">m/s</span></div>
            </div>

            <div class="gauge-metric">
                <div class="metric-title">Độ cao thân (Torso Z)</div>
                <div><span class="metric-value" id="val-z">1.02</span><span class="metric-unit">m</span></div>
            </div>

            <div class="gauge-metric">
                <div class="metric-title">Góc quay thân (Yaw)</div>
                <div><span class="metric-value" id="val-yaw">0.0</span><span class="metric-unit">deg</span></div>
            </div>

            <div class="card-header" style="margin-top: 20px;">Foot Ground Contacts</div>
            <div class="foot-indicators">
                <div class="foot-box grounded" id="foot-l">CHÂN TRÁI (L)</div>
                <div class="foot-box grounded" id="foot-r">CHÂN PHẢI (R)</div>
            </div>

            <div class="shortcut-tip">
                Phím tắt điều khiển:<br>
                <span class="kbd">W</span> Tiến | <span class="kbd">S</span> Lùi / Chậm<br>
                <span class="kbd">A</span> Rẽ trái | <span class="kbd">D</span> Rẽ phải<br>
                <span class="kbd">Space</span> Đứng thăng bằng yên
            </div>
        </div>

        <!-- Center Panel: 3D Viewport Screen -->
        <div class="viewport-wrapper">
            <div class="hud-top-bar">
                <span>HEADING: <b class="hud-compass" id="hud-heading">000° N</b></span>
                <span>STATUS: <b style="color: var(--emerald-glow);" id="hud-mode">WALKING</b></span>
            </div>

            <img class="video-stream" src="/stream.mjpg" alt="H1 Live Simulation">

            <div class="cam-bar">
                <button class="cam-btn active" onclick="setCam('chase', this)">Chase (Sau lưng)</button>
                <button class="cam-btn" onclick="setCam('orbit', this)">Orbit (Ngang)</button>
                <button class="cam-btn" onclick="setCam('front', this)">Front (Trực diện)</button>
                <button class="cam-btn" onclick="setCam('top', this)">Top-down</button>
            </div>
        </div>

        <!-- Right Panel: Teleoperation Directional Control -->
        <div class="glass-card">
            <div class="card-header">Directional Control Pad</div>

            <div class="dpad-container">
                <!-- Forward -->
                <button class="ctrl-btn" id="btn-w" onmousedown="startCmd('forward')" onmouseup="stopCmd('forward')">
                    ▲<span>TIẾN (W)</span>
                </button>
                <!-- Left & Right -->
                <div class="dpad-row">
                    <button class="ctrl-btn" id="btn-a" onmousedown="startCmd('left')" onmouseup="stopCmd('left')">
                        ◄<span>TRÁI (A)</span>
                    </button>
                    <button class="ctrl-btn" id="btn-s" onmousedown="startCmd('back')" onmouseup="stopCmd('back')">
                        ▼<span>LÙI (S)</span>
                    </button>
                    <button class="ctrl-btn" id="btn-d" onmousedown="startCmd('right')" onmouseup="stopCmd('right')">
                        ►<span>PHẢI (D)</span>
                    </button>
                </div>
                <!-- Stop / Stand -->
                <button class="ctrl-btn btn-stop" id="btn-space" onclick="toggleStand()">
                    ⏹ <span>ĐỨNG YÊN (SPACE)</span>
                </button>
            </div>

            <div class="slider-group">
                <div class="slider-label">
                    <span>Tốc độ mục tiêu (Throttle)</span>
                    <span id="speed-label">1.0 m/s</span>
                </div>
                <input type="range" id="speed-slider" min="0.2" max="1.5" step="0.1" value="1.0" oninput="updateSpeed(this.value)">
            </div>

            <div class="slider-group" style="margin-top: 18px;">
                <div class="slider-label">
                    <span>Góc xoay Camera Azimuth</span>
                    <span id="cam-label">180°</span>
                </div>
                <input type="range" id="cam-slider" min="0" max="360" step="5" value="180" oninput="updateCamAz(this.value)">
            </div>

            <button style="width: 100%; margin-top: 24px; padding: 10px; background: rgba(244, 63, 94, 0.15); border: 1px solid var(--rose-glow); color: var(--rose-glow); border-radius: 8px; font-weight: bold; cursor: pointer;" onclick="resetRobot()">
                ↻ RESET VỊ TRÍ ROBOT
            </button>
        </div>
    </div>

    <script>
        // Command state
        let currentCmd = {
            steer: 0.0,      // -1.0 (right) to +1.0 (left)
            speed: 1.0,      // 0.2 to 1.5
            mode: "walk",    // "walk" or "stand"
            cam: "chase",
            cam_az: 180.0
        };

        let keysDown = {};

        function sendCommand() {
            const url = `/cmd?mode=${currentCmd.mode}&steer=${currentCmd.steer}&speed=${currentCmd.speed}&cam=${currentCmd.cam}&cam_az=${currentCmd.cam_az}`;
            fetch(url).catch(e => {});
        }

        function startCmd(dir) {
            if (dir === 'forward') { currentCmd.mode = 'walk'; }
            if (dir === 'back') { currentCmd.speed = Math.max(0.2, currentCmd.speed - 0.2); }
            if (dir === 'left') { currentCmd.steer = 1.0; }
            if (dir === 'right') { currentCmd.steer = -1.0; }
            highlightBtn(dir, true);
            sendCommand();
        }

        function stopCmd(dir) {
            if (dir === 'left' || dir === 'right') { currentCmd.steer = 0.0; }
            highlightBtn(dir, false);
            sendCommand();
        }

        function toggleStand() {
            if (currentCmd.mode === 'stand') {
                currentCmd.mode = 'walk';
                document.getElementById('btn-space').classList.remove('active');
            } else {
                currentCmd.mode = 'stand';
                document.getElementById('btn-space').classList.add('active');
            }
            sendCommand();
        }

        function updateSpeed(val) {
            currentCmd.speed = parseFloat(val);
            document.getElementById('speed-label').innerText = val + ' m/s';
            sendCommand();
        }

        function updateCamAz(val) {
            currentCmd.cam_az = parseFloat(val);
            document.getElementById('cam-label').innerText = val + '°';
            sendCommand();
        }

        function setCam(mode, elem) {
            currentCmd.cam = mode;
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
            elem.classList.add('active');
            sendCommand();
        }

        function resetRobot() {
            fetch('/cmd?reset=true').catch(e => {});
        }

        function highlightBtn(name, active) {
            const map = { 'forward': 'btn-w', 'left': 'btn-a', 'right': 'btn-d', 'back': 'btn-s' };
            const id = map[name];
            if (id) {
                const el = document.getElementById(id);
                if (active) el.classList.add('active'); else el.classList.remove('active');
            }
        }

        // Keyboard Event Listeners
        window.addEventListener('keydown', (e) => {
            if (keysDown[e.code]) return;
            keysDown[e.code] = true;

            if (e.code === 'KeyW' || e.code === 'ArrowUp') startCmd('forward');
            if (e.code === 'KeyA' || e.code === 'ArrowLeft') startCmd('left');
            if (e.code === 'KeyD' || e.code === 'ArrowRight') startCmd('right');
            if (e.code === 'KeyS' || e.code === 'ArrowDown') startCmd('back');
            if (e.code === 'Space') { e.preventDefault(); toggleStand(); }
        });

        window.addEventListener('keyup', (e) => {
            keysDown[e.code] = false;
            if (e.code === 'KeyA' || e.code === 'ArrowLeft') stopCmd('left');
            if (e.code === 'KeyD' || e.code === 'ArrowRight') stopCmd('right');
            if (e.code === 'KeyW' || e.code === 'ArrowUp') highlightBtn('forward', false);
            if (e.code === 'KeyS' || e.code === 'ArrowDown') highlightBtn('back', false);
        });

        // Telemetry Poller (200ms)
        setInterval(async () => {
            try {
                const res = await fetch('/telemetry');
                const t = await res.json();
                document.getElementById('val-vx').innerText = (t.vx >= 0 ? '+' : '') + t.vx.toFixed(2);
                document.getElementById('val-z').innerText = t.torso_z.toFixed(2);
                document.getElementById('val-yaw').innerText = t.yaw_deg.toFixed(1);

                // Compass
                const deg = Math.round((t.yaw_deg % 360 + 360) % 360);
                const dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'];
                const dir = dirs[Math.round(deg / 45) % 8];
                document.getElementById('hud-heading').innerText = `${String(deg).padStart(3, '0')}° ${dir}`;
                document.getElementById('hud-mode').innerText = t.mode.toUpperCase();

                // Feet
                const footL = document.getElementById('foot-l');
                const footR = document.getElementById('foot-r');
                if (t.left_contact) { footL.className = 'foot-box grounded'; footL.innerText = 'CHÂN TRÁI (TIẾP ĐẤT)'; }
                else { footL.className = 'foot-box air'; footL.innerText = 'CHÂN TRÁI (NHẤC)'; }

                if (t.right_contact) { footR.className = 'foot-box grounded'; footR.innerText = 'CHÂN PHẢI (TIẾP ĐẤT)'; }
                else { footR.className = 'foot-box air'; footR.innerText = 'CHÂN PHẢI (NHẤC)'; }

            } catch (e) {}
        }, 200);
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Cyberpunk Teleoperation Cockpit for H1 Humanoid")
    parser.add_argument("--walk-ckpt", default=None, help="Path to walk checkpoint (default: autodetect latest phase2/walk)")
    parser.add_argument("--stand-ckpt", default="runs/h1_stand_v4/checkpoints/checkpoint_latest.pt", help="Path to stand checkpoint")
    parser.add_argument("--port", type=int, default=8080, help="Web Cockpit port (default: 8080)")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--width", type=int, default=640, help="Camera render width")
    parser.add_argument("--height", type=int, default=480, help="Camera render height")
    args = parser.parse_args()

    device = args.device

    # 1. Locate checkpoints
    walk_ckpt_path = args.walk_ckpt
    if walk_ckpt_path is None:
        p2_path = Path("runs/h1_phase2_walk/checkpoints/checkpoint_latest.pt")
        p1_path = Path("runs/h1_phase1_slow/checkpoints/checkpoint_latest.pt")
        if p2_path.exists():
            walk_ckpt_path = str(p2_path)
        elif p1_path.exists():
            walk_ckpt_path = str(p1_path)
        else:
            walk_ckpt_path = args.stand_ckpt

    print("\n" + "="*68)
    print("  🚀 INITIALIZING H1 CYBERPUNK TELEOPERATION COCKPIT")
    print("="*68)
    print(f"[+] Walk Checkpoint:  {walk_ckpt_path}")
    print(f"[+] Stand Checkpoint: {args.stand_ckpt}")

    # Load Walk AC
    walk_data = torch.load(walk_ckpt_path, map_location=device)
    walk_state = walk_data.get("ac_state", walk_data)
    walk_obs_dim = walk_state["actor.net.0.weight"].shape[1]
    ac_walk = ActorCritic(walk_obs_dim, 19)
    ac_walk.load_state_dict(walk_state)
    ac_walk.to(device).eval()

    # Load Stand AC
    ac_stand = None
    if Path(args.stand_ckpt).exists():
        stand_data = torch.load(args.stand_ckpt, map_location=device)
        stand_state = stand_data.get("ac_state", stand_data)
        stand_obs_dim = stand_state["actor.net.0.weight"].shape[1]
        ac_stand = ActorCritic(stand_obs_dim, 19)
        ac_stand.load_state_dict(stand_state)
        ac_stand.to(device).eval()
        print(f"[+] Stand Balance Policy Loaded: obs_dim={stand_obs_dim}")

    # Create Environment
    task_cfg = WalkTask(target_speed=1.0, max_steps=100000).to_cfg()
    env = H1Env(
        task_cfg=task_cfg,
        render_mode="rgb_array",
        frame_skip=10,
        history_len=3,
    )
    obs, _ = env.reset()

    # Shared Controller & Telemetry State
    shared = {
        "lock": threading.Lock(),
        "frame_bytes": None,
        "running": True,
        "steer": 0.0,       # -1.0 to 1.0
        "speed": 1.0,       # 0.2 to 1.5
        "mode": "walk",     # "walk" or "stand"
        "cam": "chase",     # "chase", "orbit", "front", "top"
        "cam_az": 180.0,
        "reset": False,
        "telemetry": {
            "vx": 0.0,
            "vy": 0.0,
            "torso_z": 1.02,
            "yaw_deg": 0.0,
            "left_contact": True,
            "right_contact": True,
            "mode": "walk",
        }
    }

    # HTTP & Telemetry Server
    class TeleopHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return  # silence logs

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(COCKPIT_HTML.encode("utf-8"))

            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while shared["running"]:
                        with shared["lock"]:
                            frame_data = shared["frame_bytes"]
                        if frame_data is not None:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(frame_data)).encode() + b"\r\n\r\n"
                                + frame_data + b"\r\n"
                            )
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            elif path == "/telemetry":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                with shared["lock"]:
                    data_str = json.dumps(shared["telemetry"])
                self.wfile.write(data_str.encode("utf-8"))

            elif path == "/cmd":
                params = parse_qs(parsed.query)
                with shared["lock"]:
                    if "mode" in params: shared["mode"] = params["mode"][0]
                    if "steer" in params: shared["steer"] = float(params["steer"][0])
                    if "speed" in params: shared["speed"] = float(params["speed"][0])
                    if "cam" in params: shared["cam"] = params["cam"][0]
                    if "cam_az" in params: shared["cam_az"] = float(params["cam_az"][0])
                    if "reset" in params and params["reset"][0] == "true": shared["reset"] = True

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')

    class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = ThreadedServer(("0.0.0.0", args.port), TeleopHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    print("\n" + "="*68)
    print(f"  🎮 TELEOPERATION COCKPIT READY AT:")
    print(f"  👉 http://localhost:{args.port}  (Hoặc http://<IP_Server>:{args.port})")
    print("  - Dùng phím W / A / S / D / Space trên bàn phím để lái robot!")
    print("  - Bấm Ctrl+C trong terminal để dừng.")
    print("="*68 + "\n")

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env._pelvis_bid
    camera.distance = 2.8
    camera.elevation = -12.0

    dt = env.dt
    step_count = 0

    try:
        while shared["running"]:
            step_start = time.time()

            # Read commands
            with shared["lock"]:
                steer_cmd = shared["steer"]
                speed_cmd = shared["speed"]
                mode_cmd = shared["mode"]
                cam_mode = shared["cam"]
                cam_az_cmd = shared["cam_az"]
                reset_req = shared["reset"]
                if reset_req:
                    shared["reset"] = False

            if reset_req:
                obs, _ = env.reset()
                print("[Teleop] Robot pose reset.")

            # Policy Action Selection
            if mode_cmd == "stand" and ac_stand is not None:
                # Slices out 82 features from each of the 3 history frames -> 246
                obs_stand = obs.reshape(3, 84)[:, :82].flatten()
                action, _, _ = ac_stand.act(obs_stand, device=device)
            else:
                action, _, _ = ac_walk.act(obs, device=device)
                # Modulate forward speed
                speed_scale = np.clip(speed_cmd / 1.0, 0.3, 1.4)
                action[[2, 7]] *= speed_scale  # pitch swing amplitude

                # Apply steering bias (hip yaws + torso yaw)
                # Left steer (steer > 0): rotate heading left (+yaw)
                action[0] += steer_cmd * 0.22   # left hip yaw
                action[5] += steer_cmd * 0.22   # right hip yaw
                action[10] += steer_cmd * 0.15  # torso yaw

            # Step Physics
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if terminated:
                print("  [Teleop] Robot tripped! Auto-resetting...")
                obs, _ = env.reset()

            # Heading & Camera Angle calculation
            qw, qx, qy, qz = env.data.qpos[3:7]
            yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
            yaw_deg = float(np.degrees(yaw))

            # Dynamic camera positioning
            if cam_mode == "chase":
                camera.azimuth = yaw_deg + 180.0
                camera.elevation = -12.0
                camera.distance = 2.8
            elif cam_mode == "side":
                camera.azimuth = yaw_deg + 90.0
                camera.elevation = -8.0
                camera.distance = 2.6
            elif cam_mode == "front":
                camera.azimuth = yaw_deg
                camera.elevation = -6.0
                camera.distance = 2.6
            elif cam_mode == "top":
                camera.azimuth = yaw_deg + 180.0
                camera.elevation = -60.0
                camera.distance = 3.6
            else:
                camera.azimuth = cam_az_cmd

            # Render frame
            renderer.update_scene(env.data, camera=camera)
            frame_rgb = renderer.render()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Minimal telemetry burned into image
            torso_z = float(env._get_body_z(env._torso_body))
            vx = float(env.data.qvel[0])
            vy = float(env.data.qvel[1])
            contacts = env._get_foot_contacts()

            cv2.putText(
                frame_bgr,
                f"STEER: {steer_cmd:+.1f} | SPEED: {speed_cmd:.1f} m/s | MODE: {mode_cmd.upper()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 242, 254) if mode_cmd == "walk" else (244, 63, 94),
                2,
                cv2.LINE_AA,
            )

            _, jpeg_buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

            with shared["lock"]:
                shared["frame_bytes"] = jpeg_buf.tobytes()
                shared["telemetry"] = {
                    "vx": vx,
                    "vy": vy,
                    "torso_z": torso_z,
                    "yaw_deg": yaw_deg,
                    "left_contact": bool(contacts[0] > 0.5),
                    "right_contact": bool(contacts[1] > 0.5),
                    "mode": mode_cmd,
                }

            # 50 Hz sync
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n[+] Exiting Teleoperation...")
    finally:
        shared["running"] = False
        server.shutdown()
        server.server_close()
        env.close()


if __name__ == "__main__":
    main()
