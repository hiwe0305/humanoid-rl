# Cấu hình Robot

## Unitree H1 *(Robot mặc định)*

| Thông số | Giá trị |
|---|---|
| DOF | 19 actuated joints |
| nq (position) | 26 (7 root + 19 joints) |
| nv (velocity) | 25 (6 root + 19 joints) |
| Chiều cao đứng | ~1.0m (torso) |
| Tổng trọng lượng | ~47 kg |
| Model nguồn | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) |

### Joints & PD Gains

| Joint | kp (N·m/rad) | kd (N·m·s/rad) | Torque limit (N·m) | Range (rad) |
|---|---|---|---|---|
| hip_yaw (×2) | 150 | 5.0 | 200 | [-0.43, 0.43] |
| hip_roll (×2) | 150 | 5.0 | 200 | [-0.43, 0.43] |
| hip_pitch (×2) | 200 | 5.0 | 200 | [-1.57, 1.57] |
| knee (×2) | 200 | 5.0 | 300 | [-0.26, 2.05] |
| ankle (×2) | 60 | 1.5 | 40 | [-0.87, 0.52] |
| torso | 200 | 5.0 | 200 | [-2.35, 2.35] |
| shoulder_pitch (×2) | 60 | 1.5 | 40 | [-2.87, 2.87] |
| shoulder_roll (×2) | 60 | 1.5 | 40 | varies |
| shoulder_yaw (×2) | 60 | 1.5 | 40 | varies |
| elbow (×2) | 60 | 1.5 | 40 | [-1.25, 2.61] |

### Default Standing Pose

```python
H1_DEFAULT_QPOS = [
    # Left leg: yaw, roll, hip_pitch, knee, ankle
    0.0, 0.0, -0.35, 0.70, -0.35,
    # Right leg: yaw, roll, hip_pitch, knee, ankle
    0.0, 0.0, -0.35, 0.70, -0.35,
    # Torso
    0.0,
    # Left arm: shoulder_pitch, roll, yaw, elbow
    0.0, 0.3, 0.0, 0.3,
    # Right arm
    0.0, -0.3, 0.0, 0.3,
]
```

Đây là tư thế đứng tối ưu với gối hơi cong (0.70 rad) để ổn định hơn tư thế thẳng hoàn toàn.

### Body Names

```python
torso_body  = "torso_link"
pelvis_body = "pelvis"
left_foot   = "left_ankle_link"
right_foot  = "right_ankle_link"
```

### File cấu hình

```
humanoid_rl/robots/unitree_h1.py
```

### Download model

```python
pip install robot_descriptions
from robot_descriptions import h1_mj_description
print(h1_mj_description.MJCF_PATH)
# → ~/.cache/robot_descriptions/mujoco_menagerie/unitree_h1/h1.xml
```

---

## MuJoCo Built-in Humanoid *(Legacy)*

> ⚠️ Chỉ dùng cho test nhanh. Production nên dùng Unitree H1.

| Thông số | Giá trị |
|---|---|
| DOF | 17 actuated joints |
| nq | 24 |
| nv | 23 |
| Model nguồn | Embedded XML trong `robots/mujoco_humanoid.py` |

### Khi nào dùng

- Unit tests (`tests/test_envs.py`)
- Debug nhanh không cần download
- Benchmark baseline

### Chuyển sang H1

Thay:
```python
from scripts.benchmark import make_env_fn
env_fn = make_env_fn("humanoid", "walk")  # old
```

Bằng:
```python
from humanoid_rl.envs.mujoco.h1_env import H1Env
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
env_fn = lambda: H1Env(WalkTask().to_cfg())  # new
```

---

## Thêm robot mới

1. Tạo file `humanoid_rl/robots/<robot_name>.py`:

```python
def get_robot_cfg() -> dict:
    return {
        "name":               "my_robot",
        "mjcf_path_absolute": "/path/to/robot.xml",
        "dof":                12,
        "torso_body_name":    "base_link",
        "pelvis_body_name":   "pelvis",
        "left_foot_body":     "left_foot",
        "right_foot_body":    "right_foot",
        "min_torso_height":   0.4,
        "reset_noise_scale":  0.01,
        "kp": 100.0,
        "kd": 5.0,
    }
```

2. Tạo `H1Env` subclass hoặc dùng `MuJoCoHumanoidEnv` với cfg mới.
