# Kiến trúc Hệ thống

## Tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING LOOP                               │
│                                                                 │
│  ┌──────────┐    collect    ┌──────────────┐   update          │
│  │  PPO     │ ───────────▶  │  VecEnv      │ ─────────────┐   │
│  │  Trainer │               │  (8–32 envs) │              │   │
│  └──────────┘               └──────────────┘              │   │
│       ▲                           │ obs,reward             │   │
│       │                           ▼                        │   │
│       │                    ┌──────────────┐                │   │
│       │                    │  H1Env       │                │   │
│       │                    │  (MuJoCo)    │                │   │
│       │                    └──────────────┘                │   │
│       │                                                     │   │
│  ┌────┴──────────────────────────────────────────────────┐ │   │
│  │  PPORolloutBuffer (numpy, RAM)                        │◀┘   │
│  │  4,096 transitions/update (8 envs × 512 steps)       │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  ┌───────────────────────────────────────────────────────┐     │
│  │  ActorCritic (PyTorch, CUDA)                          │     │
│  │  Actor: [74]→[256]→[128]→[19]                        │     │
│  │  Critic: [74]→[256]→[128]→[1]                        │     │
│  └───────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Hybrid C++/Python

```
Python (điều phối)              C++ (hiệu năng)
─────────────────               ───────────────
PPO training loop           →   RolloutBuffer (lock-free MPSC)
DataCollector               →   HDF5Writer (chunked GZIP)
VecEnv wrapper                  Bindings: pybind11 → numpy
```

**Lý do hybrid:** Python GIL làm chậm I/O. C++ xử lý ring buffer và disk write không bị GIL block.

## Luồng dữ liệu

```
env.reset()
    │
    ▼
obs [74 dims]  ─────────────────────────────────────────▶  ActorCritic
                                                                │
                                              action (19 dims)  │
                                                     log_prob   │
                                                     value V(s) │
                                                                ▼
obs ──▶  PD Controller ──▶ torques [19] ──▶ MuJoCo physics
                              │
                    frame_skip=10 steps
                              │
                              ▼
                   next_obs, reward, done
                              │
                              ▼
                    PPORolloutBuffer.add()
```

## Observation Space (74 dims)

| Index | Nội dung | Dim | Mô tả |
|---|---|---|---|
| 0–23 | `qpos[2:]` | 24 | Root z + quaternion + 19 joint positions |
| 24–48 | `qvel` | 25 | Root lin/ang velocity + 19 joint velocities |
| 49–52 | body heights | 4 | torso, pelvis, left_ankle, right_ankle z |
| 53–54 | foot contacts | 2 | Binary left/right ground contact |
| 55–73 | `prev_action` | 19 | Previous joint targets (proprioception) |

> **Tại sao bỏ root x?** Robot luôn di chuyển theo x → không có thông tin relative. Policy cần velocity (đã có), không cần absolute position.

## Action Space (19 dims)

Action = joint position offsets từ default standing pose, normalized [-1, 1].

```
a ∈ [-1, 1]^19
q_target = q_default + a × 0.4 rad
τ = kp × (q_target - q) + kd × (0 - dq)
```

**Lý do dùng PD thay vì direct torque:**
- Smoother motion (implicit damping)
- Sim-to-real: real H1 hardware dùng PD control
- Easier to tune (kp, kd được set theo phần cứng thật)

## Control Frequency

| Tham số | Giá trị |
|---|---|
| MuJoCo timestep | 0.002s (500 Hz) |
| frame_skip | 10 |
| Control dt | 0.02s (50 Hz) |
| Episode length (walk) | 1000 steps = 20s |

## Reward Architecture

```
R_total = R_locomotion + R_stability + R_energy

R_locomotion = w_vel × exp(-(vx - v_target)²)
             - w_lat × vy²
             + w_height × Δz          (stair)
             + w_vz × max(vz, 0)      (jump)

R_stability  = w_upright × z_torso_vec
             + alive_bonus × [torso_z > min_h]

R_energy     = -w_torque × Σ τᵢ²
             - w_rate × Σ (aᵢ - aᵢ₋₁)²
```

## Module Dependencies

```
scripts/
  train_quick.py
      └── PPOTrainer
              ├── SyncVecEnv / SubprocVecEnv
              │       └── H1Env
              │               ├── PDController
              │               ├── Terrain (injected XML)
              │               ├── DomainRandomizer
              │               ├── PushRandomizer
              │               └── Rewards (Loco + Stab + Energy)
              ├── ActorCritic (PyTorch)
              ├── PPORolloutBuffer
              ├── Logger (TB + WandB)
              └── Checkpoint
```
