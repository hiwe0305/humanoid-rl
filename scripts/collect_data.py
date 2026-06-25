"""
scripts/collect_data.py
Dataset collection script.

Usage:
    # Collect 500 episodes using random policy:
    python scripts/collect_data.py robot=humanoid task=walk collector.n_episodes=500

    # Collect with a trained policy:
    python scripts/collect_data.py robot=h1 task=run \
        collector.policy_checkpoint=runs/ppo/checkpoints/checkpoint_latest.pt \
        collector.n_episodes=1000
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import hydra
from omegaconf import DictConfig
import numpy as np
import torch

from scripts.train import _build_env_fn


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("  Humanoid-RL Data Collection")
    print("=" * 60)

    env_fn = _build_env_fn(cfg)

    # Optionally load policy
    policy_fn = None
    checkpoint_path = cfg.get("collector", {}).get("policy_checkpoint", None)

    if checkpoint_path:
        print(f"[Collect] Loading policy from {checkpoint_path}")
        from humanoid_rl.utils.checkpoint import Checkpoint
        from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic

        # Create temp env to get dims
        tmp_env = env_fn()
        obs_dim = tmp_env.observation_space.shape[0]
        act_dim = tmp_env.action_space.shape[0]
        tmp_env.close()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ac = ActorCritic(obs_dim, act_dim).to(device)
        ckpt = Checkpoint("").load(checkpoint_path)
        ac.load_state_dict(ckpt["ac_state"])
        ac.eval()

        @torch.no_grad()
        def policy_fn(obs: np.ndarray):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            actions, log_probs, _, values = ac.get_action_and_value(obs_t)
            return (
                actions.cpu().numpy(),
                log_probs.cpu().numpy(),
                values.squeeze(-1).cpu().numpy(),
            )

        print("[Collect] Using trained policy")
    else:
        print("[Collect] Using random policy (no checkpoint specified)")

    collector_cfg = cfg.get("collector", {})
    n_episodes    = int(collector_cfg.get("n_episodes", 200))
    n_envs        = int(collector_cfg.get("n_envs", 8))
    output_dir    = str(collector_cfg.get("output_dir", "data/raw"))

    from humanoid_rl.data.collector import DataCollector
    collector = DataCollector(
        env_fn     = env_fn,
        n_envs     = n_envs,
        output_dir = output_dir,
        robot      = cfg.robot.name,
        task       = cfg.task.name,
    )

    output_path = collector.collect(
        policy_fn  = policy_fn,
        n_episodes = n_episodes,
        save_every = int(collector_cfg.get("save_every", 50)),
    )
    collector.close()

    # Print dataset stats
    from humanoid_rl.data.hdf5_dataset import HDF5Dataset
    ds = HDF5Dataset(str(output_path), mode="r")
    stats = ds.stats()
    ds.close()

    print("\n[Dataset Stats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
