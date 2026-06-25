# Roadmap Cải thiện

## Trạng thái hiện tại

### ✅ Đã hoàn thành (Phase 1)

| Component | Status | Mô tả |
|---|---|---|
| MuJoCo H1 env | ✅ Production | PD control, domain rand, terrain |
| Terrain generator | ✅ 6 loại | flat, stair, rough, slope, hurdle, gap |
| Domain randomization | ✅ | Mass, friction, damping, gear, push |
| PPO algorithm | ✅ | Clipped, value clip, entropy, GAE |
| HDF5 dataset | ✅ | Episodic, chunked GZIP |
| C++ extensions | ✅ | RolloutBuffer, HDF5Writer |
| Training pipeline | ✅ | Hydra + standalone |
| Tests (9/9) | ✅ | Env, VecEnv, Dataset, C++ |

---

## 🔴 High Priority

### 1. SubprocVecEnv Auto-selection

**Vấn đề:** `SyncVecEnv` không tận dụng 64 cores khi n_envs > 8.

**Fix:**
```python
# humanoid_rl/algorithms/ppo/ppo.py
def _make_vec_env(self, env_fn, n_envs):
    threshold = 8
    if n_envs > threshold:
        from humanoid_rl.utils.vec_env_wrapper import SubprocVecEnv
        return SubprocVecEnv([env_fn] * n_envs)
    else:
        from humanoid_rl.utils.vec_env_wrapper import SyncVecEnv
        return SyncVecEnv([env_fn] * n_envs)
```

**Impact:** 3–5× throughput tăng với n_envs=32.

### 2. Curriculum Learning Manager

**Vấn đề:** Phải train từng task thủ công, không có auto-progression.

**Thiết kế:**
```python
class CurriculumManager:
    stages = [
        ("walk", "flat",   100_000, {"target_return": 500}),
        ("walk", "rough",  200_000, {"target_return": 400}),
        ("run",  "rough",  500_000, {"target_return": 300}),
        ("stair","stair",  500_000, {"target_return": 200}),
    ]

    def advance(self, current_return):
        """Tự động chuyển stage khi đạt target."""
```

### 3. Proprioceptive History (Multi-frame obs)

**Vấn đề:** Hiện tại H1Env hỗ trợ `history_len` nhưng chưa được dùng.

**Khi dùng:** `history_len=3` — stack 3 frame obs để xử lý latency và partial obs.

```bash
python scripts/train_quick.py --history-len 3  # cần thêm arg này
```

---

## 🟡 Medium Priority

### 4. Reference Motion Tracking (AMP-style)

**Mục đích:** Học motion đẹp hơn bằng cách bắt chước motion capture data.

**Thư viện tham khảo:** [PHC](https://github.com/ZhengyiLuo/PHC), [AMP](https://arxiv.org/abs/2104.02180)

**Cần:** Motion capture data (BVH/SMPLx format) + discriminator network

### 5. Sim-to-Real Transfer Pipeline

**Checklist:**
- [x] PD position control (done)
- [x] Domain randomization (done)
- [ ] Latency simulation (`action_delay=1 step`)
- [ ] Sensor noise injection
- [ ] Motor model (backlash, hysteresis)
- [ ] Real robot inference script (`scripts/deploy.py`)

```python
# Thêm action delay trong env
class H1EnvWithDelay(H1Env):
    def __init__(self, delay_steps=1, **kwargs):
        super().__init__(**kwargs)
        self._action_buffer = deque(maxlen=delay_steps + 1)
```

### 6. Better Observation

**Thiếu hiện tại:**
- `q - q_default`: relative joint position từ standing pose
- IMU: linear acceleration, angular velocity (tách riêng từ qvel)
- Terrain height scan (raycast): 17 điểm xung quanh chân

**Observation đầy đủ (target):**
```
74 dims (hiện tại)
+ 19 (q_relative)
+ 17 (terrain height scan)
= 110 dims
```

### 7. SAC / TD-MPC2 Alternative

**Lý do:** SAC ổn định hơn PPO cho continuous control, không cần on-policy data.

```python
from humanoid_rl.algorithms.sac.sac import SACTrainer  # chưa có
trainer = SACTrainer(env_fn, replay_buffer_size=1_000_000)
```

---

## 🟢 Long-term

### 8. Isaac Lab / IsaacGym Integration

**Lợi ích:** GPU-accelerated physics → 100,000+ steps/sec

**Khó khăn:** Cần NVIDIA GPU + Isaac SDK riêng biệt

### 9. Multi-task Learning

**Mục đích:** 1 policy cho tất cả tasks (walk/run/stair/jump)

```python
# Task embedding vào observation
task_id = one_hot([0, 1, 2, 3])  # walk/run/stair/jump
obs = concat([env_obs, task_id])
```

### 10. Policy Distillation

**Mục đích:** 4 task-specific policies → 1 universal policy (teacher-student)

---

## Estimation thời gian

| Improvement | Effort | Impact |
|---|---|---|
| SubprocVecEnv auto | 1 ngày | 🔴 High |
| Curriculum manager | 2 ngày | 🔴 High |
| Action delay | 0.5 ngày | 🟡 Medium |
| Sensor noise | 0.5 ngày | 🟡 Medium |
| Terrain height scan | 2 ngày | 🟡 Medium |
| AMP/reference motion | 1 tuần | 🟢 Long-term |
| SAC implementation | 1 tuần | 🟢 Long-term |
| Isaac Lab | 2 tuần | 🟢 Long-term |
