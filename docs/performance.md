# Performance & Tối ưu hóa

## Benchmark đo được (máy hiện tại)

**Hardware:** Intel 64-core CPU + NVIDIA RTX 3060 + 128GB RAM

| Metric | Giá trị |
|---|---|
| MuJoCo stepping (single env) | **5,676 steps/sec** |
| SyncVecEnv (16 envs, H1) | **1,898 steps/sec** |
| PPO training throughput | **~700 FPS** (CUDA) |
| Dataset collection (8 envs) | **24.8 episodes/sec** |

## Phân tích bottleneck

```
Total training time per update:
  Env stepping (n_envs=8, n_steps=512):  ~5.5s   (85%)
  PPO update (GPU):                       ~0.9s   (14%)
  I/O / logging:                          ~0.1s   (1%)
```

→ **Bottleneck là simulation**, không phải GPU. Cần nhiều envs.

## Tối ưu simulation

### 1. Tăng n_envs (tác động lớn nhất)

```bash
# Server 64 cores: dùng 32-48 envs
python scripts/train_quick.py --task walk --n-envs 32
```

Lý thuyết: 32 envs → ~3,600 FPS (so với ~700 FPS với 8 envs)

### 2. SubprocVecEnv cho n_envs > 16

`SyncVecEnv` (hiện tại) chạy tuần tự trong 1 process. Cho n_envs lớn, cần subprocess:

```python
# Trong PPOTrainer._make_vec_env():
from humanoid_rl.utils.vec_env_wrapper import SubprocVecEnv

def _make_vec_env(self, env_fn, n_envs):
    if n_envs <= 8:
        from humanoid_rl.utils.vec_env_wrapper import SyncVecEnv
        return SyncVecEnv([env_fn] * n_envs)
    else:
        return SubprocVecEnv([env_fn] * n_envs)
```

### 3. Giảm frame_skip (cân nhắc)

| frame_skip | Control Hz | steps/sec (single) | Stability |
|---|---|---|---|
| 5 | 100 Hz | 5,676 | Cần torque smoother |
| 10 | 50 Hz | ~8,000 | ✅ H1 default |
| 20 | 25 Hz | ~11,000 | Quá coarse |

H1 dùng `frame_skip=10` (50 Hz) vì hardware H1 thật chạy ở 50 Hz.

### 4. Render mode

```python
# Tắt hoàn toàn renderer khi training
env = H1Env(task_cfg, render_mode=None)  # không tạo renderer

# Chỉ bật khi cần
env = H1Env(task_cfg, render_mode="rgb_array")  # cho video
```

## Tối ưu GPU

### Batch size

```python
# RTX 3060 (12GB VRAM):
ppo_cfg = {
    "n_envs":     16,
    "n_steps":    1024,
    "batch_size": 512,   # ~optimal cho 12GB
    "n_epochs":   8,
}
```

### Mixed Precision (Optional)

```python
# Trong actor_critic.py — thêm autocast:
from torch.cuda.amp import autocast

with autocast():
    action, log_prob, entropy, value = ac.get_action_and_value(obs_t)
```

### Torch Compile (PyTorch 2.x)

```python
# Sau khi tạo actor-critic:
self.ac = torch.compile(self.ac, mode="reduce-overhead")
```

## Memory

| Component | RAM usage |
|---|---|
| Rollout buffer (16 envs × 2048 steps × 74 obs) | ~75 MB |
| Model (108K params, float32) | ~0.4 MB |
| H1 model (MuJoCo) | ~20 MB/env |
| Total (16 envs) | ~400 MB |

## Profiling

```bash
# Python profiler
PYTHONPATH=. python -m cProfile -o profile.out scripts/train_quick.py --total-steps 5000
python -m pstats profile.out

# Line profiler (cài: pip install line_profiler)
kernprof -l -v scripts/train_quick.py

# MuJoCo stepping benchmark
python scripts/benchmark.py --robot h1 --n-envs 1 --steps 10000
python scripts/benchmark.py --robot h1 --n-envs 32 --steps 1000
```

## Scaling guide

| Server | Recommended config |
|---|---|
| Laptop (4 cores, no GPU) | n_envs=4, CPU, n_steps=256 |
| Workstation (16 cores, RTX 3060) | n_envs=8–16, CUDA |
| Server (64 cores, A100) | n_envs=48, CUDA, SubprocVecEnv |
| Multi-GPU | Không được hỗ trợ mặc định, cần thêm `torch.nn.DataParallel` |
