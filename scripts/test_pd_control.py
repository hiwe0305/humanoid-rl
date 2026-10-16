"""
scripts/test_pd_control.py
PD Control verification for Unitree H1.

Tests PD controller independently from RL to verify:
  - PD holds default pose (half-sitting)
  - Random actions don't cause immediate collapse
  - PD gains compute correct torques
  - Joint stiffness under gravity

Reference: humanoid-gym and LearningHumanoidWalking both use NO balance
force. They rely on half-sitting default pose + RL reward shaping.
The PD controller holds joint targets; RL learns to output actions
that maintain balance.

Usage:
    PYTHONPATH=. MUJOCO_GL=egl python scripts/test_pd_control.py
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import mujoco
from humanoid_rl.envs.mujoco.h1_env import (
    H1Env, H1_DEFAULT_QPOS, H1_KP, H1_KD, H1_ACTION_SCALE,
)
from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"

def header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─── Test 1: Half-sitting pose — PD holds squat posture ───────────────────

def test_squat_pose():
    """With half-sitting default pose, PD should hold squat position.
    Roll must stay near 0 (wide base = laterally stable).
    Pitch will drift (inverted pendulum) but robot shouldn't collapse fast."""
    header("Test 1: Half-Sitting Pose (PD Holds Squat)")
    env = H1Env(StandTask().to_cfg(), render_mode=None)
    obs, info = env.reset(seed=42)

    n_steps = 50
    torso_heights, pitches, rolls, knee_angles = [], [], [], []

    for step in range(n_steps):
        obs, _, term, trunc, info = env.step(np.zeros(19, dtype=np.float32))
        d = env.data
        torso_z = float(env._get_body_z("torso_link"))
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        lk = float(d.qpos[7+3])  # left knee
        rk = float(d.qpos[7+8])  # right knee

        torso_heights.append(torso_z)
        pitches.append(pitch)
        rolls.append(roll)
        knee_angles.append([lk, rk])

        if term:
            print(f"  Terminated at step {step}")
            break

    n = len(torso_heights)
    pitches = np.array(pitches)
    rolls = np.array(rolls)
    knee_angles = np.array(knee_angles)
    torso_heights = np.array(torso_heights)

    # Check first 10 steps (before pitch drift dominates)
    first10_pitch = np.max(np.abs(pitches[:min(10, n)]))
    first10_roll = np.max(np.abs(rolls[:min(10, n)]))
    knee_avg = np.mean(np.abs(np.degrees(knee_angles)))
    avg_knee_deg = np.mean(np.degrees(knee_angles))

    print(f"  Steps survived: {n}/{n_steps}")
    print(f"  First 10 steps: max|pitch|={np.degrees(first10_pitch):.1f}°, max|roll|={np.degrees(first10_roll):.1f}°")
    print(f"  Avg knee angle: {knee_avg:.1f}° (should be ~34° = half-sitting)")
    print(f"  Torso height: {torso_heights[0]:.3f} → {torso_heights[-1]:.3f}")

    # Pass criteria: roll stays near 0, knee holds squat, survived >20 steps
    roll_ok = first10_roll < np.radians(5)  # roll < 5° in first 10 steps
    knee_ok = 25 < knee_avg < 50  # knee around 34° (half-sitting)
    survive_ok = n > 20
    print(f"  Roll stable: {PASS if roll_ok else FAIL} (< 5° in first 10 steps)")
    print(f"  Knee squat:  {PASS if knee_ok else FAIL} (25-50° avg)")
    print(f"  Survival:    {PASS if survive_ok else FAIL} (> 20 steps)")

    env.close()
    return roll_ok and knee_ok and survive_ok


# ─── Test 2: Sinusoidal gait — doesn't immediately crash ──────────────────

def test_sinusoidal_gait():
    """CPG gait should produce forward motion without immediate crash."""
    header("Test 2: Sinusoidal Gait (CPG Walking)")
    env = H1Env(WalkTask(target_speed=1.0).to_cfg(), render_mode=None)
    obs, info = env.reset(seed=0)

    ctrl_dt = env._ctrl_dt
    freq = 2.0
    phase = 0.0
    torso_heights, pitches, velocities = [], [], []

    for step in range(200):
        phase += ctrl_dt * freq * 2 * np.pi
        sl, sr = np.sin(phase), np.sin(phase + np.pi)
        amp = 0.5

        action = np.zeros(19, dtype=np.float32)
        action[2]  = amp * sl * 0.8       # L hip pitch
        action[7]  = amp * sr * 0.8       # R hip pitch
        action[3]  = abs(sl) * amp * 0.6  # L knee
        action[8]  = abs(sr) * amp * 0.6  # R knee
        action[4]  = -sl * amp * 0.3      # L ankle
        action[9]  = -sr * amp * 0.3      # R ankle
        action[11] = -amp * sl * 0.3      # L arm
        action[15] = -amp * sr * 0.3      # R arm
        action = np.clip(action, -1.0, 1.0)

        obs, _, term, trunc, info = env.step(action)
        d = env.data
        torso_z = float(env._get_body_z("torso_link"))
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))

        torso_heights.append(torso_z)
        pitches.append(pitch)
        velocities.append(float(d.qvel[0]))

        if term or trunc:
            break

    n = len(torso_heights)
    velocities = np.array(velocities)
    pitches = np.array(pitches)

    avg_vx = np.mean(velocities)
    max_vx = np.max(velocities)
    max_pitch = np.max(np.abs(pitches))

    print(f"  Steps survived: {n}/200")
    print(f"  Max vx:     {max_vx:+.3f} m/s")
    print(f"  Max |pitch|: {np.degrees(max_pitch):.1f}°")
    print(f"  Forward motion: {PASS if max_vx > 0.05 else FAIL}")
    print(f"  Survival:       {PASS if n > 30 else FAIL}")

    env.close()
    return n > 30


# ─── Test 3: Small random actions — no immediate collapse ─────────────────

def test_random_actions():
    """Small random actions should not cause immediate collapse.
    Robot should survive >15 steps with random joint perturbations."""
    header("Test 3: Small Random Actions (No Collapse)")
    env = H1Env(StandTask().to_cfg(), render_mode=None)
    obs, info = env.reset(seed=42)

    rng = np.random.default_rng(seed=123)
    amplitude = 0.1

    n_steps = 50
    torso_heights, rolls = [], []

    for step in range(n_steps):
        action = rng.uniform(-amplitude, amplitude, 19).astype(np.float32)
        obs, _, term, trunc, info = env.step(action)
        d = env.data
        torso_z = float(env._get_body_z("torso_link"))
        qw, qx, qy, qz = d.qpos[3:7]
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))

        torso_heights.append(torso_z)
        rolls.append(roll)

        if term:
            print(f"  Terminated at step {step}")
            break

    n = len(torso_heights)
    rolls = np.array(rolls)
    max_roll = np.max(np.abs(rolls))
    avg_knee = np.mean(np.abs(np.degrees(env.data.qpos[7+3])))  # left knee

    print(f"  Steps survived: {n}/{n_steps}")
    print(f"  Max |roll|:  {np.degrees(max_roll):.1f}°")
    print(f"  Knee angle:  {avg_knee:.1f}°")

    roll_ok = max_roll < np.radians(10)  # roll stays under 10°
    survive_ok = n > 15
    print(f"  Roll stable: {PASS if roll_ok else FAIL} (< 10°)")
    print(f"  Survival:    {PASS if survive_ok else FAIL} (> 15 steps)")

    env.close()
    return roll_ok and survive_ok


# ─── Test 4: PD gain verification ─────────────────────────────────────────

def test_pd_gains():
    """Verify PD gains are properly configured."""
    header("Test 4: PD Gain Verification")
    env = H1Env(StandTask().to_cfg(), render_mode=None)
    obs, info = env.reset(seed=0)

    pd = env._pd
    results = []

    print(f"  KP range: [{pd.kp.min():.0f}, {pd.kp.max():.0f}] Nm/rad")
    print(f"  KD range: [{pd.kd.min():.0f}, {pd.kd.max():.0f}] Nm*s/rad")
    print(f"  Action scale: {pd.action_scale:.2f} rad")
    print(f"  Default pose (knee): {np.degrees(pd._q_default[3]):.1f}°")

    # Test: max action on knee -> expected torque
    action = np.zeros(19, dtype=np.float32)
    action[3] = 1.0
    q = env.data.qpos.copy()
    dq = np.zeros_like(env.data.qvel)
    torques = pd.compute_torques(action, q, dq)
    gear = pd._gear[3]
    expected = pd.kp[3] * pd.action_scale / gear
    torque_ok = abs(torques[3] - expected) < 5.0
    print(f"  Knee torque: got={torques[3]:.1f}, expected={expected:.1f}")
    print(f"  Torque calc: {PASS if torque_ok else FAIL}")
    results.append(torque_ok)

    # Zero torque at default pose
    q_default = np.concatenate([np.zeros(7), pd._q_default])
    torques_zero = pd.compute_torques(np.zeros(19), q_default, np.zeros(25))
    zero_ok = np.max(np.abs(torques_zero)) < 0.1
    print(f"  Zero torque at default: {PASS if zero_ok else FAIL}")
    results.append(zero_ok)

    # PD direction check
    obs, _ = env.reset(seed=0)
    action = np.zeros(19, dtype=np.float32)
    action[3] = 0.5
    obs, _, term, _, _ = env.step(action)
    knee_angle = float(env.data.qpos[7+3])
    torque_sign_ok = knee_angle > 0
    print(f"  Knee angle after command: {np.degrees(knee_angle):.1f}°")
    print(f"  PD direction: {PASS if torque_sign_ok else FAIL}")
    results.append(torque_sign_ok)

    env.close()
    return all(results)


# ─── Test 5: Gravity resistance (joint stiffness) ────────────────────────

def test_gravity_resistance():
    """PD joints should resist gravity — maintain squat posture."""
    header("Test 5: Gravity Resistance (Joint Stiffness)")
    env = H1Env(StandTask().to_cfg(), render_mode=None)
    obs, info = env.reset(seed=0)

    # Let settle
    for _ in range(50):
        obs, _, term, _, _ = env.step(np.zeros(19))
        if term:
            break
    q_settled = env.data.qpos[7:].copy()

    # Run 150 more steps
    heights, pitches, rolls = [], [], []
    for _ in range(150):
        obs, _, term, _, _ = env.step(np.zeros(19))
        d = env.data
        torso_z = float(env._get_body_z("torso_link"))
        qw, qx, qy, qz = d.qpos[3:7]
        pitch = np.arctan2(2*(qw*qy - qz*qx), 1 - 2*(qx**2 + qy**2))
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        heights.append(torso_z)
        pitches.append(pitch)
        rolls.append(roll)
        if term:
            break

    heights = np.array(heights)
    q_final = env.data.qpos[7:].copy()

    max_drift = np.max(np.abs(q_final - q_settled))
    knee_drift = np.abs(np.degrees(q_final[3] - q_settled[3]))
    max_roll = np.max(np.abs(np.array(rolls)))

    print(f"  Knee drift: {knee_drift:.1f}°")
    print(f"  Max joint drift: {np.degrees(max_drift):.1f}°")
    print(f"  Height std: {np.std(heights):.4f}m")
    print(f"  Max |roll|: {np.degrees(max_roll):.1f}°")

    # Half-sitting: knee should stay near 34° (half-sitting = 0.6 rad)
    knee_stable = knee_drift < 5.0  # knee drift < 5°
    roll_stable = max_roll < np.radians(5)  # roll < 5°
    print(f"  Knee stable: {PASS if knee_stable else FAIL} (< 5° drift)")
    print(f"  Roll stable: {PASS if roll_stable else FAIL} (< 5°)")

    env.close()
    return knee_stable and roll_stable


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Unitree H1 - PD Control Verification")
    print("  (No balance force — like humanoid-gym & LearningHumanoidWalking)")
    print("="*60)

    tests = [
        ("Squat Pose",          test_squat_pose),
        ("Sinusoidal Gait",     test_sinusoidal_gait),
        ("Random Actions",      test_random_actions),
        ("PD Gains",            test_pd_gains),
        ("Gravity Resistance",  test_gravity_resistance),
    ]

    results = {}
    for name, fn in tests:
        try:
            ok = fn()
            results[name] = ok
        except Exception as e:
            print(f"\n  {FAIL} EXCEPTION: {e}")
            import traceback; traceback.print_exc()
            results[name] = False

    header("SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {PASS if ok else FAIL}  {name}")
    print(f"\n  {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
