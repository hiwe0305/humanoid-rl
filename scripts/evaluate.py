"""
scripts/evaluate.py
Policy evaluation script.
Runs a trained policy for N episodes and reports statistics.

Usage:
    python scripts/evaluate.py \\
        --checkpoint runs/ppo/checkpoints/checkpoint_latest.pt \\
        --robot humanoid --task walk \\
        --n-episodes 20 \\
        --record-video          # save mp4 of first episode
"""

import sys, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--robot",       default="humanoid", choices=["humanoid","h1"])
    parser.add_argument("--task",        default="walk",     choices=["walk","run","stair","jump"])
    parser.add_argument("--n-episodes",  type=int, default=10)
    parser.add_argument("--record-video",action="store_true")
    parser.add_argument("--video-path",  default="eval_rollout.mp4")
    args = parser.parse_args()

    # Build env
    from scripts.benchmark import make_env_fn
    env = make_env_fn(args.robot, args.task)()

    # Load policy
    from humanoid_rl.algorithms.ppo.actor_critic import ActorCritic
    from humanoid_rl.utils.checkpoint import Checkpoint

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    ac = ActorCritic(obs_dim, act_dim).to(device)
    ckpt = Checkpoint("").load(args.checkpoint)
    ac.load_state_dict(ckpt["ac_state"])
    ac.eval()
    print(f"[Eval] Loaded policy | obs_dim={obs_dim} act_dim={act_dim} device={device}")
    print(f"[Eval] Robot={args.robot} | Task={args.task} | Episodes={args.n_episodes}")

    ep_returns = []
    ep_lengths = []
    frames     = []

    for ep in range(args.n_episodes):
        obs, _ = env.reset(seed=ep)
        ep_ret  = 0.0
        ep_len  = 0
        record_this = args.record_video and ep == 0

        while True:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _, _ = ac.get_action_and_value(obs_t)
            action_np = action.squeeze(0).cpu().numpy()

            if record_this:
                frame = env.render()  # requires render_mode='rgb_array'
                if frame is not None:
                    frames.append(frame)

            obs, reward, term, trunc, _ = env.step(action_np)
            ep_ret += reward
            ep_len  += 1
            if term or trunc:
                break

        ep_returns.append(ep_ret)
        ep_lengths.append(ep_len)
        print(f"  Episode {ep+1:>3}/{args.n_episodes}: return={ep_ret:>8.2f}  len={ep_len}")

    env.close()

    print(f"\n{'='*45}")
    print(f"  Mean return : {np.mean(ep_returns):>8.2f} ± {np.std(ep_returns):.2f}")
    print(f"  Min / Max   : {np.min(ep_returns):>8.2f} / {np.max(ep_returns):.2f}")
    print(f"  Mean length : {np.mean(ep_lengths):>8.1f}")
    print(f"{'='*45}")

    if frames:
        import imageio
        imageio.mimwrite(args.video_path, frames, fps=50)
        print(f"\n[Eval] Video saved: {args.video_path}")


if __name__ == "__main__":
    main()
