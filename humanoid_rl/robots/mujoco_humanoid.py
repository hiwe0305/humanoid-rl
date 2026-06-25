"""
humanoid_rl/robots/mujoco_humanoid.py
MuJoCo built-in humanoid model configuration.
Using inline MJCF XML string (MuJoCo 3.x has no bundled humanoid.xml in pip package).
"""
from __future__ import annotations

# Compact but physically reasonable humanoid model
# Adapted from dm_control / MuJoCo reference humanoid
HUMANOID_XML = """
<mujoco model="humanoid">
  <compiler angle="degree" inertiafromgeom="true"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba=".8 .6 .4 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <option integrator="RK4" timestep="0.002"/>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" name="floor" pos="0 0 0" size="20 20 .125" type="plane" material="MatPlane" condim="3"/>
    <body name="torso" pos="0 0 1.4">
      <camera name="track" mode="trackcom" pos="0 -3 1" xyaxes="1 0 0 0 0 1"/>
      <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
      <geom fromto="0 -.07 0 0 .07 0" name="torso1" size=".07" type="capsule"/>
      <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
      <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size=".06" type="capsule"/>
      <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
        <geom fromto="0 -.06 0 0 .06 0" name="waist" size=".0625" type="capsule"/>
        <joint axis="0 0 1" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
        <joint axis="1 0 0" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="20" type="hinge"/>
        <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
          <joint axis="0 1 0" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
          <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size=".09" type="capsule"/>
          <body name="right_thigh" pos="0 -0.1 -0.04">
            <joint axis="1 0 0" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
            <joint axis="0 0 1" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
            <joint axis="0 1 0" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
            <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size=".06" type="capsule"/>
            <body name="right_shin" pos="0 0.01 -0.403">
              <joint axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
              <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size=".049" type="capsule"/>
              <body name="right_foot" pos="0 0 -0.45">
                <geom name="right_foot" pos="0 0 0" size=".075" type="sphere" user="0"/>
              </body>
            </body>
          </body>
          <body name="left_thigh" pos="0 0.1 -0.04">
            <joint axis="-1 0 0" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
            <joint axis="0 0 -1" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
            <joint axis="0 1 0" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
            <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size=".06" type="capsule"/>
            <body name="left_shin" pos="0 -0.01 -0.403">
              <joint axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
              <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size=".049" type="capsule"/>
              <body name="left_foot" pos="0 0 -0.45">
                <geom name="left_foot" pos="0 0 0" size=".075" type="sphere" user="0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="right_upper_arm" pos="0 -0.17 0.06">
        <joint axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
        <joint axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
        <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size=".04 .16" type="capsule"/>
        <body name="right_lower_arm" pos=".18 -.18 -.18">
          <joint axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
          <geom fromto=".01 .01 .01 .17 .17 .17" name="right_larm" size=".031" type="capsule"/>
          <geom name="right_hand" pos=".18 .18 .18" size=".04" type="sphere"/>
        </body>
      </body>
      <body name="left_upper_arm" pos="0 0.17 0.06">
        <joint axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
        <joint axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
        <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size=".04 .16" type="capsule"/>
        <body name="left_lower_arm" pos=".18 .18 -.18">
          <joint axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
          <geom fromto=".01 -.01 .01 .17 -.17 .17" name="left_larm" size=".031" type="capsule"/>
          <geom name="left_hand" pos=".18 -.18 .18" size=".04" type="sphere"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="left_hipknee" range="-0.5 0.5">
      <joint coef="-1" joint="left_knee"/>
    </fixed>
    <fixed name="right_hipknee" range="-0.5 0.5">
      <joint coef="-1" joint="right_knee"/>
    </fixed>
  </tendon>
  <actuator>
    <motor gear="40" joint="abdomen_y" name="abdomen_y"/>
    <motor gear="40" joint="abdomen_z" name="abdomen_z"/>
    <motor gear="40" joint="abdomen_x" name="abdomen_x"/>
    <motor gear="40" joint="right_hip_x" name="right_hip_x"/>
    <motor gear="40" joint="right_hip_z" name="right_hip_z"/>
    <motor gear="120" joint="right_hip_y" name="right_hip_y"/>
    <motor gear="80" joint="right_knee" name="right_knee"/>
    <motor gear="40" joint="left_hip_x" name="left_hip_x"/>
    <motor gear="40" joint="left_hip_z" name="left_hip_z"/>
    <motor gear="120" joint="left_hip_y" name="left_hip_y"/>
    <motor gear="80" joint="left_knee" name="left_knee"/>
    <motor gear="20" joint="right_shoulder1" name="right_shoulder1"/>
    <motor gear="20" joint="right_shoulder2" name="right_shoulder2"/>
    <motor gear="40" joint="right_elbow" name="right_elbow"/>
    <motor gear="20" joint="left_shoulder1" name="left_shoulder1"/>
    <motor gear="20" joint="left_shoulder2" name="left_shoulder2"/>
    <motor gear="40" joint="left_elbow" name="left_elbow"/>
  </actuator>
  <asset>
    <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" name="sky" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1=".8 .6 .4" rgb2=".8 .6 .4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
</mujoco>
"""

# Save path for loading
import os
import tempfile

_HUMANOID_TMP_PATH: str | None = None


def get_humanoid_xml_path() -> str:
    """Write the humanoid XML to a temp file and return its path."""
    global _HUMANOID_TMP_PATH
    if _HUMANOID_TMP_PATH is None or not os.path.exists(_HUMANOID_TMP_PATH):
        tmp = tempfile.NamedTemporaryFile(
            suffix=".xml", prefix="humanoid_", delete=False, mode="w"
        )
        tmp.write(HUMANOID_XML)
        tmp.flush()
        _HUMANOID_TMP_PATH = tmp.name
        tmp.close()
    return _HUMANOID_TMP_PATH


MUJOCO_HUMANOID_CFG = {
    "name":               "mujoco_humanoid",
    "mjcf_file":          None,
    "mjcf_path_absolute": None,   # populated lazily via get_humanoid_xml_path()
    "default_torque_limit": 100.0,
    "min_torso_height":   0.5,
    "reset_noise_scale":  0.01,
    "torso_body_name":    "torso",
    "foot_geom_names":    ["left_foot", "right_foot"],
    "kp": 100.0,
    "kd": 10.0,
}


def get_humanoid_cfg(overrides: dict | None = None) -> dict:
    """Get MuJoCo built-in humanoid config with lazy XML path creation."""
    cfg = dict(MUJOCO_HUMANOID_CFG)
    cfg["mjcf_path_absolute"] = get_humanoid_xml_path()
    if overrides:
        cfg.update(overrides)
    return cfg
