# Dataset Pipeline

## Tổng quan

```
Policy (trained) hoặc Random
         │
         ▼ (actions)
 VecEnv (8 envs parallel)
         │
         ▼ (obs, action, reward, done)
   DataCollector
         │
         ▼
  HDF5Dataset (.hdf5)
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
HDF5Dataset          HDF5TransitionDataset
(sequential          (random batch sampling
 episodes)            cho Offline RL)
```

## Thu dataset

```bash
# Thu 1000 episodes với random policy (exploration data)
python scripts/collect_data.py robot=h1 task=walk collector.n_episodes=1000

# Thu với trained policy
python scripts/collect_data.py robot=h1 task=walk \
    collector.policy_checkpoint=runs/h1/walk/checkpoints/checkpoint_final.pt \
    collector.n_episodes=2000
```

## Cấu trúc HDF5

```
dataset.hdf5
├── metadata/
│   ├── robot       "h1"
│   ├── task        "walk"
│   ├── n_episodes  1000
│   └── total_steps 847293
│
├── episode_0000/
│   ├── observations  [T × 74]  float32   obs từng bước
│   ├── actions       [T × 19]  float32   action (joint targets)
│   ├── rewards       [T]       float32   reward mỗi bước
│   ├── dones         [T]       bool      episode end flag
│   ├── next_obs      [T × 74]  float32   obs sau khi step
│   ├── values        [T]       float32   V(s) từ critic
│   └── log_probs     [T]       float32   log π(a|s)
│
├── episode_0001/
│   └── ...
```

**Nén:** Chunked GZIP (level 4) — tiết kiệm ~60% dung lượng so với uncompressed.

## Dùng dataset

### Sequential (Behavior Cloning)

```python
from humanoid_rl.data.hdf5_dataset import HDF5Dataset

ds = HDF5Dataset("data/raw/h1_walk_123456.hdf5", mode="r")
stats = ds.stats()
print(f"Episodes: {stats['n_episodes']}")
print(f"Mean return: {stats['mean_return']:.2f}")

# Đọc episode
ep = ds.get_episode(0)
obs     = ep["observations"]  # numpy [T, 74]
actions = ep["actions"]       # numpy [T, 19]
rewards = ep["rewards"]       # numpy [T]
ds.close()
```

### Random sampling (Offline RL)

```python
from humanoid_rl.data.hdf5_dataset import HDF5TransitionDataset
from torch.utils.data import DataLoader

dataset = HDF5TransitionDataset("data/raw/h1_walk_123456.hdf5")
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

for batch in loader:
    obs      = batch["obs"]       # [256, 74]
    action   = batch["action"]    # [256, 19]
    reward   = batch["reward"]    # [256]
    next_obs = batch["next_obs"]  # [256, 74]
    done     = batch["done"]      # [256]
    # → Dùng cho IQL, TD3+BC, ...
```

## DataCollector API

```python
from humanoid_rl.data.collector import DataCollector

collector = DataCollector(
    env_fn     = env_fn,       # factory function
    n_envs     = 8,            # parallel envs
    output_dir = "data/raw",   # output directory
    robot      = "h1",
    task       = "walk",
)

# Thu với random policy
output = collector.collect(n_episodes=500, save_every=100)

# Thu với trained policy
import torch
from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic

ac = ActorCritic(74, 19).to("cuda")
ac.load_state_dict(torch.load("checkpoint.pt")["ac_state"])

def policy_fn(obs):
    obs_t = torch.as_tensor(obs, device="cuda")
    action, log_prob, _, value = ac.get_action_and_value(obs_t)
    return action.cpu().numpy(), log_prob.cpu().numpy(), value.squeeze(-1).cpu().numpy()

output = collector.collect(policy_fn=policy_fn, n_episodes=1000)
collector.close()
```

## Tốc độ thu dataset

| Config | Speed |
|---|---|
| 1 env, random policy | ~800 steps/sec |
| 8 envs, random policy | ~5,600 steps/sec |
| 8 envs, trained policy (GPU) | ~4,200 steps/sec |

## Use cases

| Use case | Tool | Mô tả |
|---|---|---|
| **On-policy RL** | `PPOTrainer` trực tiếp | Không cần lưu data, PPO tự thu |
| **Behavior Cloning** | `HDF5Dataset` sequential | Học từ demo tốt |
| **Offline RL (IQL)** | `HDF5TransitionDataset` | Tái sử dụng data cũ |
| **Data Augmentation** | Đọc + transform | Random flip trái/phải, noise |
| **Analysis** | `ds.stats()` | Kiểm tra chất lượng data |
