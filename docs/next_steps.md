# Next Steps — Humanoid RL Roadmap

## Current Status (June 2026)

### ✅ Fixed
- **PD position control** added to generic humanoid env — đứng vững với zero action
- **Default config** switched to `robot: h1` (có PD control sẵn)
- **Foot contact detection** body-based (chính xác)
- **CoM observation** + root orientation tracking
- **CurriculumManager** — 5 phase auto-progression
- **Isaac Sim scaffold** — directory structure + shared interface
- **Video recording** — view_env.py + training video recorder
- **21/21 tests pass**

### ⚠️ Environment constraints
- **Không có NVIDIA GPU** — chỉ chạy MuJoCo CPU
- **Không có display** — cần `MUJOCO_GL=egl` cho offscreen render
- **No Isaac Sim installed** — scaffold sẵn sàng chờ GPU/Isaac

---

## Next Steps

### 1. [NOW] Khởi động training cơ bản

```bash
source venv/bin/activate

# View robot standing (video)
MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --robot h1 --task stand --record --steps 200

# Chạy curriculum training (có video tự động)
MUJOCO_GL=egl PYTHONPATH=. python scripts/train_curriculum.py --robot h1 --n-envs 16
```

### 2. [1-2 ngày] Hoàn thiện training pipeline

| Task | File | Chi tiết |
|---|---|---|
| SubprocVecEnv auto (n_envs > 8) | `ppo.py` | Auto switch SyncVecEnv → SubprocVecEnv |
| Curriculum checkpoint resume | `train_curriculum.py` | Resume đúng phase |
| Hydra config integration | `train_curriculum.py` | Dùng config yaml thay vì hardcode |

### 3. [1-2 ngày] Nâng cấp observation

| Component | Current | Target |
|---|---|---|
| Proprioceptive history | H1 có history_len=3 | Generic humanoid cũng history_len=3 |
| q - q_default relative joints | Missing | Add 19 dims |
| Terrain height scan | Missing | 17 raycast points |
| Target obs dim | 82 per frame | ~110 per frame |

### 4. [3-4 ngày] Sim-to-real features

| Feature | File | Mức độ |
|---|---|---|
| Action delay (latency) | `h1_env.py` | 0.5 ngày |
| Sensor noise injection | `domain_rand.py` | 0.5 ngày |
| Motor model | `pd_control.py` | 1 ngày |
| Deploy script | `scripts/deploy.py` | 1 ngày |

### 5. [Khi có GPU] Isaac Sim Integration

**Prerequisites:**
- NVIDIA GPU (RTX 4090+ recommended)
- Isaac Sim installed
- H1 URDF/USD file

**Implementation steps:**
1. Implement `IsaacH1Env` methods (`_get_obs`, `_step_simulation`, `_reset_model`)
2. Verify same observation dims (246 for H1 with history_len=3)
3. Test policy transfer: train on MuJoCo → deploy on Isaac
4. GPU-accelerated training: 100K+ steps/s

### 6. [1 tuần] Advanced algorithms

| Algorithm | Why |
|---|---|
| SAC | Off-policy, more sample efficient |
| TD-MPC2 | Model-based, SOTA for locomotion |
| AMP | Motion imitation from reference data |

---

## Viewing the Robot

```bash
# 1. Record video of H1 standing (offline)
MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --robot h1 --record --steps 200

# 2. Interactive viewer (cần display)
MUJOCO_GL=glfw PYTHONPATH=. python scripts/view_env.py --robot h1

# 3. Record walking gait
MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --robot h1 --task walk --speed 0.5 --record --steps 500

# 4. Play with manual CPG control 
MUJOCO_GL=glfw PYTHONPATH=. python scripts/play_h1.py
# Hoặc record video:
MUJOCO_GL=egl PYTHONPATH=. python scripts/play_h1.py --offscreen --steps 500 --output output/h1_demo.mp4

# 5. Dùng generic humanoid
MUJOCO_GL=egl PYTHONPATH=. python scripts/view_env.py --robot humanoid --record --steps 200
```

## Training with Video Recording

```bash
# Curriculum training with auto video recording
MUJOCO_GL=egl PYTHONPATH=. python scripts/train_curriculum.py --robot h1 --n-envs 16

# Videos saved to: runs/curriculum/phase_*/videos/
```

---

## Architecture Reference

```
                              ┌──────────────────────┐
                              │   BaseHumanoidEnv    │
                              │  (abstract interface)│
                              └──────┬───────────────┘
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
        │  MuJoCoH1Env    │ │ MuJoCoHumanoid  │ │  IsaacH1Env     │
        │  (H1, PD, GPU)  │ │ (generic, PD)   │ │ (GPU-accel)     │
        └─────────────────┘ └─────────────────┘ └─────────────────┘
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                            ┌─────────────────┐
                            │   PPOTrainer    │
                            │  (shared algo)  │
                            └─────────────────┘
                                      │
                            ┌─────────────────┐
                            │ VideoRecorder   │
                            │ (training viz)  │
                            └─────────────────┘
```

## Quick Commands Reference

| Command | Purpose |
|---|---|
| `scripts/view_env.py --robot h1 --record` | Record H1 standing/walking video |
| `scripts/play_h1.py --offscreen` | Manual CPG control + video |
| `scripts/train_curriculum.py` | Full curriculum training |
| `scripts/verify_h1_physics.py` | Verify robot physics |
| `pytest tests/ -v` | Run all tests |

## KPI Tracking

| Metric | Current | Target |
|---|---|---|
| Tests passing | 21/21 | 25+ |
| Stand reward (100 steps) | ~12 | 30+ |
| Walk steps/sec | ~5K (CPU) | 100K+ (GPU Isaac) |
| Curriculum phases | 5 | Auto-advance |
