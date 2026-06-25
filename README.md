# Humanoid-RL

**High-performance Reinforcement Learning platform for humanoid robot locomotion.**

Trains the Unitree H1 humanoid robot (19 DOF) using Proximal Policy Optimization (PPO)
with a hybrid C++/Python architecture for maximum throughput.

---

## Tính năng chính

| Tính năng | Mô tả |
|---|---|
| 🤖 **Robot** | Unitree H1 (19 DOF, 25 DOF velocity) — official MuJoCo Menagerie model |
| 🎮 **Control** | PD position control (sim-to-real ready) |
| 🏔️ **Terrain** | flat / rough / stair / slope / hurdle / gap |
| 🎲 **Domain Rand** | Mass, friction, damping, gear, push perturbations |
| ⚡ **Performance** | ~5,000+ steps/sec (single env), CUDA-accelerated PPO |
| 💾 **Dataset** | HDF5 episodic storage (C++ writer, GZIP compressed) |
| 📊 **Logging** | TensorBoard + WandB |

## Cấu trúc thư mục

```
humanoid-rl/
├── csrc/                    C++ high-performance core
│   ├── envpool/             Lock-free rollout buffer
│   └── io/                  HDF5 writer (chunked + GZIP)
├── humanoid_rl/
│   ├── robots/              Robot configurations
│   ├── envs/mujoco/         MuJoCo environments + terrain + PD control
│   ├── rewards/             Modular reward functions
│   ├── algorithms/ppo/      PPO Actor-Critic
│   ├── data/                Dataset collection pipeline
│   └── utils/               VecEnv, Logger, Checkpoint
├── configs/                 Hydra YAML configs
├── scripts/                 Entry points (train, collect, benchmark, viz)
├── docs/                    📚 This documentation
└── tests/                   Unit + integration tests
```

## Cài đặt nhanh

```bash
git clone <repo> && cd humanoid-rl
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install robot_descriptions  # auto-download H1 model

# Build C++ extensions
cd build && cmake .. && make -j16 && cd ..
```

## Chạy nhanh

```bash
source venv/bin/activate
PYTHONPATH=. MUJOCO_GL=egl

# Training H1 walk (với domain randomization):
python scripts/train_quick.py --task walk --n-envs 8

# Các task khác:
python scripts/train_quick.py --task stair    # leo cầu thang
python scripts/train_quick.py --task run      # chạy nhanh
python scripts/train_quick.py --task jump     # nhảy qua vật cản

# Thu dataset:
python scripts/collect_data.py robot=h1 task=walk

# Benchmark tốc độ:
python scripts/benchmark.py --robot h1 --n-envs 16

# Render video:
python scripts/visualize.py --robot h1 --task walk \
    --checkpoint runs/h1/walk/checkpoints/checkpoint_final.pt \
    --mode video --output h1_walk.mp4
```

## Tài liệu chi tiết

| File | Nội dung |
|---|---|
| [architecture.md](architecture.md) | Kiến trúc toàn hệ thống |
| [robots.md](robots.md) | Cấu hình robot (H1, MuJoCo Humanoid) |
| [tasks.md](tasks.md) | Các task + terrain + reward design |
| [training.md](training.md) | Hướng dẫn training (PPO, hyperparams) |
| [dataset.md](dataset.md) | Pipeline thu và xử lý dataset |
| [performance.md](performance.md) | Benchmark + tối ưu hóa |
| [improvements.md](improvements.md) | Roadmap cải thiện |
