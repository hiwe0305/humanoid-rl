"""
humanoid_rl/robots/unitree_h1.py
Unitree H1 humanoid robot configuration.
Uses the official MJCF from MuJoCo Menagerie (Google DeepMind).
"""
from __future__ import annotations
from pathlib import Path


def _get_h1_xml_path() -> str:
    """
    Try robot_descriptions package first, then pip cache, then local copy.
    "/"""
    # Try robot_descriptions (recommended, requires internet)
    try:
        from robot_descriptions import h1_mj_description
        return h1_mj_description.MJCF_PATH
    except ImportError:
        pass

    # Try pip cache
    cache_roots = [
        Path.home() / ".cache" / "robot_descriptions" / "mujoco_menagerie" / "unitree_h1",
        Path("/home/ubuntu/.cache/robot_descriptions/mujoco_menagerie/unitree_h1"),
    ]
    for d in cache_roots:
        xml = d / "h1.xml"
        if xml.exists():
            return str(xml.resolve())

    # Fallback: local assets
    local = Path(__file__).parent / "assets" / "unitree_h1.xml"
    if local.exists():
        return str(local)

    raise FileNotFoundError(
        f"Unitree H1 MJCF not found. "
        f"Tried: robot_descriptions, {cache_roots}, {local}"
    )


# Real body/geom names from MuJoCo Menagerie H1 model
UNITREE_H1_CFG = {
    "name":                 "h1",
    "mjcf_file":            None,
    "mjcf_path_absolute":   None,   # lazy: call get_h1_cfg()
    # H1 specs (nq=26, nv=25, nu=19)
    "dof":                  19,
    "default_torque_limit": 200.0,   # Nm (H1 has strong actuators)
    "min_torso_height":     0.6,     # m — H1 standing=1.06m; 0.3 was too low (robot survives lying flat)
    "reset_noise_scale":    0.01,
    # Body names (from MuJoCo Menagerie)
    "torso_body_name":      "torso_link",
    "pelvis_body_name":     "pelvis",
    "left_foot_body":       "left_ankle_link",
    "right_foot_body":      "right_ankle_link",
    # PD gains (default for position control)
    "kp": 200.0,
    "kd": 10.0,
    # Joint names (19 actuators)
    "joint_names": [
        "left_hip_yaw_joint",   "left_hip_roll_joint",   "left_hip_pitch_joint",
        "left_knee_joint",      "left_ankle_joint",
        "right_hip_yaw_joint",  "right_hip_roll_joint",  "right_hip_pitch_joint",
        "right_knee_joint",     "right_ankle_joint",
        "torso_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",   "left_elbow_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",   "right_elbow_joint",
    ],
}


def get_h1_cfg(overrides: dict | None = None) -> dict:
    """Get Unitree H1 config with lazy MJCF path resolution."""
    cfg = dict(UNITREE_H1_CFG)
    cfg["mjcf_path_absolute"] = _get_h1_xml_path()
    if overrides:
        cfg.update(overrides)
    return cfg
