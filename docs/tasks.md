# Tasks & Terrain

## Tổng quan các Task

| Task | Terrain | Speed | Max steps | Đặc điểm |
|---|---|---|---|---|
| **walk** | flat / rough | 1.0 m/s | 1000 | Task cơ bản, học trước |
| **run** | rough | 3.0 m/s | 1000 | Tốc độ cao, địa hình gồ ghề |
| **stair** | stair (6 bước) | 0.5 m/s | 2000 | Leo cầu thang, thưởng height gain |
| **jump** | hurdle (4 cọc) | 1.5 m/s | 1500 | Nhảy qua vật cản, thưởng vertical vel |

---

## Walk Task

**Mục tiêu:** Duy trì vận tốc 1.0 m/s về phía trước, giữ thẳng đứng.

```python
from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
task = WalkTask(target_speed=1.0, terrain="flat")
```

**Terrain:** Flat (mặc định) hoặc rough (địa hình gồ ghề nhẹ)

**Reward:**
```
R = 2.0 × exp(-(vx - 1.0)²)     # forward velocity
  - 0.5 × vy²                     # lateral penalty
  + 1.5 × z_upright               # upright posture
  + 1.0 × alive                   # alive bonus
  - 0.0005 × Σ τᵢ²               # energy efficiency
  - 0.001 × Σ (aᵢ - aᵢ₋₁)²      # smoothness
```

**Curriculum:** Walk → Walk rough → Run rough

---

## Run Task

**Mục tiêu:** Duy trì 3.0 m/s trên địa hình gồ ghề.

```python
from humanoid_rl.envs.mujoco.tasks.run_task import RunTask
task = RunTask(target_speed=3.0, terrain="rough")
```

**Terrain:** `rough` — 30 khối ngẫu nhiên cao 2–12cm

```
 ___  _ ___    ___
|   ||_|   |__|   |__ flat_floor
```

**Reward đặc biệt:** `forward_vel_weight=3.0` (cao hơn walk) để ưu tiên tốc độ.

---

## Stair Task

**Mục tiêu:** Leo cầu thang 6 bậc, mỗi bậc cao 15cm.

```python
from humanoid_rl.envs.mujoco.tasks.stair_task import StairTask
task = StairTask(n_steps=6, step_height=0.15)
```

**Terrain:** `stair` — 6 bậc thang từ vị trí x=2m

```
           ┌───┐
       ┌───┘   │
   ┌───┘       │
───┘           │  (leo từ trái sang phải)
x=2m          x=4.4m
```

**Reward đặc biệt:**
```
height_gain_weight = 3.0   # thưởng nhiều khi tăng độ cao
vertical_vel_weight = 0.5   # thưởng vận tốc đi lên
upright_weight = 2.0        # quan trọng hơn khi leo cầu thang
```

**Tham số terrain:**

| Param | Default | Mô tả |
|---|---|---|
| `n_steps` | 6 | Số bậc thang |
| `step_height` | 0.15m | Chiều cao mỗi bậc |
| `step_depth` | 0.40m | Chiều sâu mỗi bậc |
| `step_width` | 2.0m | Chiều rộng bậc thang |
| `start_x` | 2.0m | Vị trí bắt đầu cầu thang |

---

## Jump Task

**Mục tiêu:** Nhảy qua 4 cọc cản cao 25cm.

```python
from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask
task = JumpTask(n_hurdles=4, hurdle_height=0.25)
```

**Terrain:** `hurdle` — thanh ngang cao 25cm, cách nhau 2.5m

```
   |      |      |      |
   |      |      |      |
───┴──────┴──────┴──────┴──
x=2m    4.5m   7.0m   9.5m
```

**Reward đặc biệt:**
```
vertical_vel_weight = 2.0   # thưởng vận tốc đứng (nhảy lên)
height_gain_weight  = 1.0   # thưởng khi bay lên
torque_penalty      = 0.0002  # nhỏ hơn → cho phép lực lớn khi bật nhảy
```

---

## Terrain Generator API

```python
from humanoid_rl.envs.mujoco.terrain import get_terrain_xml, inject_terrain

# Lấy XML snippet
xml = get_terrain_xml("stair", n_steps=8, step_height=0.12)

# Inject vào MJCF
modified_xml = inject_terrain(base_xml, "rough", n_blocks=50, seed=123)
```

**Tất cả terrain types:**

```python
terrain_types = ["flat", "stair", "rough", "slope", "hurdle", "gap"]
```

### Tùy chỉnh terrain

```python
# Cầu thang cao hơn
task = StairTask(n_steps=8, step_height=0.20)

# Địa hình rất gồ ghề
from humanoid_rl.envs.mujoco.h1_env import H1Env
env = H1Env(
    WalkTask().to_cfg(),
    terrain_type="rough",
    terrain_kwargs={"n_blocks": 60, "max_height": 0.25, "seed": 42}
)
```

---

## Thêm task mới

```python
# humanoid_rl/envs/mujoco/tasks/my_task.py
from dataclasses import dataclass, field

@dataclass
class MyTask:
    target_speed: float = 1.5
    max_steps:    int   = 1000
    terrain:      str   = "slope"     # terrain type

    rewards: dict = field(default_factory=lambda: {
        "locomotion": {"forward_vel_weight": 2.0, ...},
        "stability":  {"upright_weight": 1.5, ...},
        "energy":     {"torque_penalty": 0.001, ...},
    })

    def to_cfg(self) -> dict:
        return {
            "target_speed": self.target_speed,
            "max_steps":    self.max_steps,
            "terrain":      self.terrain,
            "rewards":      self.rewards,
        }
```
