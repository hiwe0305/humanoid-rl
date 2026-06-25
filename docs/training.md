# Hướng dẫn Training

## Training cơ bản

```bash
source venv/bin/activate
PYTHONPATH=. MUJOCO_GL=egl

# Walk (bắt đầu từ đây)
python scripts/train_quick.py --task walk --n-envs 8

# Run (sau khi walk converge)
python scripts/train_quick.py --task run --n-envs 8

# Stair (với terrain thật)
python scripts/train_quick.py --task stair --n-envs 8

# Jump
python scripts/train_quick.py --task jump --n-envs 8
```

## PPO Hyperparameters

```python
ppo_cfg = {
    # Rollout
    "n_envs":        8,       # số env song song (tăng lên 32+ cho server)
    "n_steps":       512,     # steps thu mỗi update
    "batch_size":    256,     # mini-batch size
    "n_epochs":      4,       # số pass qua buffer

    # Discount
    "gamma":         0.99,    # discount factor
    "gae_lambda":    0.95,    # GAE lambda

    # PPO clip
    "clip_coef":     0.2,     # clipping epsilon
    "clip_value":    True,    # clip value loss cũng

    # Loss weights
    "value_coef":    0.5,     # critic loss weight
    "entropy_coef":  0.005,   # entropy bonus (khuyến khích exploration)

    # Optimization
    "lr":            3e-4,    # learning rate
    "max_grad_norm": 1.0,     # gradient clipping
    "lr_anneal":     True,    # linear LR decay to 0

    # Total training
    "total_steps":   10_000_000,  # production: 10M steps
    "save_every":    10,          # save checkpoint mỗi N updates
}
```

## Network Architecture

```
Actor (Policy):
  Input [74] → Linear(256) → LayerNorm → ELU
             → Linear(128) → LayerNorm → ELU
             → Linear(19)  [action mean]
  log_std: learned parameter (not input-dependent)
  → Normal(action_mean, exp(log_std))

Critic (Value function):
  Input [74] → Linear(256) → LayerNorm → ELU
             → Linear(128) → LayerNorm → ELU
             → Linear(1)   [V(s)]
```

- **Orthogonal init:** Tất cả linear layers, output layer với gain=0.01 (actor) / 1.0 (critic)
- **Online obs normalization:** Welford algorithm, cập nhật rolling mean/std trong quá trình collect

## Checkpoint

Checkpoints được lưu trong `runs/<experiment>/checkpoints/`:

```
runs/
  h1/
    walk/
      checkpoints/
        checkpoint_step_40960.pt
        checkpoint_step_81920.pt
        checkpoint_final.pt
```

**Load checkpoint:**
```python
from humanoid_rl.utils.checkpoint import Checkpoint
state = Checkpoint("").load("runs/h1/walk/checkpoints/checkpoint_final.pt")
# state contains: ac_state, optimizer_state, global_step, update_count, cfg
```

## Domain Randomization

Được bật mặc định trong `train_quick.py`:

```python
DOMAIN_RAND_CFG = {
    "mass_ratio":     0.15,   # mass ±15%
    "friction_ratio": 0.25,   # friction ±25%
    "damping_ratio":  0.15,   # damping ±15%
    "gear_ratio":     0.08,   # actuator strength ±8%
}

PUSH_RAND_CFG = {
    "push_prob":      0.003,  # 0.3% mỗi step
    "push_force_max": 60.0,   # N
    "push_duration":  5,      # steps
}
```

**Tắt domain rand:**
```bash
python scripts/train_quick.py --task walk --no-rand --no-push
```

## Curriculum Learning (khuyến nghị)

Thứ tự training tối ưu:

```
1. Walk flat (100k steps)  →  agent học đi cơ bản
       ↓
2. Walk rough (200k steps) →  xử lý địa hình gồ ghề
       ↓
3. Run rough (500k steps)  →  tốc độ cao
       ↓
4. Stair (500k steps)      →  leo cầu thang
       ↓
5. Jump hurdle (500k steps)→  nhảy qua vật cản
```

Load policy từ task trước để khởi đầu:

```python
trainer = PPOTrainer(env_fn=env_fn, cfg=ppo_cfg, log_dir="runs/h1/stair")
trainer.load("runs/h1/run/checkpoints/checkpoint_final.pt")  # warm start
trainer.train()
```

## Theo dõi Training

**TensorBoard:**
```bash
tensorboard --logdir runs/
```

**Metrics quan trọng:**
| Metric | Giải thích | Tốt khi |
|---|---|---|
| `mean_ep_return` | Trung bình reward mỗi episode | Tăng dần |
| `loss_pi` | Policy loss | Tiến về 0 |
| `loss_v` | Value loss | Giảm dần |
| `approx_kl` | KL divergence | < 0.02 |
| `clip_fraction` | % samples bị clip | 0.1–0.3 |
| `fps` | Samples/second | Càng cao càng tốt |

## Production Training

Với server 64 cores + GPU:

```python
ppo_cfg = {
    "n_envs":       32,         # tận dụng nhiều cores
    "n_steps":      2048,       # longer rollout
    "batch_size":   2048,       # larger batch
    "n_epochs":     10,         # more updates
    "total_steps":  10_000_000, # 10M steps
    ...
}
```

Dùng `SubprocVecEnv` cho n_envs > 16:
```python
# Trong PPOTrainer._make_vec_env:
from humanoid_rl.utils.vec_env_wrapper import SubprocVecEnv
return SubprocVecEnv([env_fn for _ in range(n_envs)])
```
