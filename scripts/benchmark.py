"""
scripts/benchmark.py
Performance benchmark: measure environment stepping throughput.

Usage:
    source venv/bin/activate
    python scripts/benchmark.py
    python scripts/benchmark.py --robot h1 --n-envs 32 --steps 5000
"""

import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np


def make_env_fn(robot: str, task: str):
    """Build env factory based on robot name."""
    if robot == "humanoid":
        from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
        robot_cfg = get_humanoid_cfg()
    elif robot == "h1":
        from humanoid_rl.robots.unitree_h1 import get_h1_cfg
        robot_cfg = get_h1_cfg()
    else:
        raise ValueError(f"Unknown robot: {robot}")

    task_map = {
        "walk":  "humanoid_rl.envs.mujoco.tasks.walk_task.WalkTask",
        "run":   "humanoid_rl.envs.mujoco.tasks.run_task.RunTask",
        "jump":  "humanoid_rl.envs.mujoco.tasks.jump_task.JumpTask",
        "stair": "humanoid_rl.envs.mujoco.tasks.stair_task.StairTask",
    }
    import importlib
    mod_path, cls_name = task_map[task].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_path), cls_name)
    task_obj = cls()
    task_cfg = task_obj.to_cfg()

    xml_path = robot_cfg["mjcf_path_absolute"]

    def env_fn():
        import mujoco, numpy as np
        from gymnasium import spaces
        from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
        from humanoid_rl.envs.base_env import BaseHumanoidEnv
        from humanoid_rl.rewards.locomotion import LocomotionRewards
        from humanoid_rl.rewards.stability  import StabilityRewards
        from humanoid_rl.rewards.energy     import EnergyRewards

        class QuickEnv(MuJoCoHumanoidEnv):
            def __init__(self):
                BaseHumanoidEnv.__init__(self, robot_cfg, task_cfg, None, 5)
                self._width = 64; self._height = 64
                self._model = mujoco.MjModel.from_xml_path(xml_path)
                self._data  = mujoco.MjData(self._model)
                self._sim_dt  = self._model.opt.timestep
                self._ctrl_dt = self._sim_dt * 5
                obs_dim = self._compute_obs_dim()
                act_dim = self._model.nu
                self._observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                self._action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
                tl = self._model.actuator_forcerange[:, 1].copy()
                self._torque_limits = tl if not np.all(tl == 0) else np.ones(act_dim) * 100.0
                rwd = task_cfg.get("rewards", {})
                self._loco_rewards   = LocomotionRewards(rwd.get("locomotion", {}))
                self._stab_rewards   = StabilityRewards(rwd.get("stability", {}))
                self._energy_rewards = EnergyRewards(rwd.get("energy", {}))
                self._max_steps    = task_cfg.get("max_steps", 1000)
                self._target_speed = task_cfg.get("target_speed", 1.0)
                self._renderer = None; self._viewer = None
                self._torso_body   = robot_cfg.get("torso_body_name",  "torso")
                self._pelvis_body  = robot_cfg.get("pelvis_body_name", "pelvis")
                self._lfoot_body   = robot_cfg.get("left_foot_body",   "left_foot")
                self._rfoot_body   = robot_cfg.get("right_foot_body",  "right_foot")
                self._init_qpos = self._data.qpos.copy()
                self._init_qvel = self._data.qvel.copy()
        return QuickEnv()
    return env_fn


def run_benchmark(robot: str = "humanoid",
                  task: str = "walk",
                  n_envs: int = 16,
                  n_steps: int = 2000,
                  use_subproc: bool = False) -> dict:
    """
    Run stepping benchmark and return stats.
    Returns dict with keys: robot, task, n_envs, n_steps,
                             total_steps, elapsed_sec, steps_per_sec
    """
    env_fn = make_env_fn(robot, task)

    print(f"\n{'='*55}")
    print(f"  Benchmark: robot={robot} | task={task}")
    print(f"  n_envs={n_envs} | n_steps={n_steps} | subproc={use_subproc}")
    print(f"{'='*55}")

    if use_subproc:
        from humanoid_rl.utils.vec_env_wrapper import SubprocVecEnv
        envs = SubprocVecEnv([env_fn for _ in range(n_envs)])
    else:
        from humanoid_rl.utils.vec_env_wrapper import SyncVecEnv
        envs = SyncVecEnv([env_fn for _ in range(n_envs)])

    obs, _ = envs.reset()
    act_dim = envs.act_dim

    # Warmup
    for _ in range(10):
        actions = np.random.uniform(-1, 1, (n_envs, act_dim)).astype(np.float32)
        envs.step(actions)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_steps):
        actions = np.random.uniform(-1, 1, (n_envs, act_dim)).astype(np.float32)
        envs.step(actions)
    elapsed = time.perf_counter() - t0

    total_steps   = n_steps * n_envs
    steps_per_sec = total_steps / elapsed

    print(f"  Total env-steps : {total_steps:>10,}")
    print(f"  Elapsed         : {elapsed:>10.2f} s")
    print(f"  Steps / second  : {steps_per_sec:>10,.0f}")
    print(f"  ms / step       : {elapsed/total_steps*1000:>10.4f} ms")

    envs.close()
    return {
        "robot":         robot,
        "task":          task,
        "n_envs":        n_envs,
        "n_steps":       n_steps,
        "total_steps":   total_steps,
        "elapsed_sec":   round(elapsed, 3),
        "steps_per_sec": round(steps_per_sec, 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Humanoid-RL stepping benchmark")
    parser.add_argument("--robot",     default="humanoid", choices=["humanoid", "h1"])
    parser.add_argument("--task",      default="walk",     choices=["walk","run","stair","jump"])
    parser.add_argument("--n-envs",    type=int, default=16)
    parser.add_argument("--steps",     type=int, default=2000)
    parser.add_argument("--subproc",   action="store_true", help="Use SubprocVecEnv")
    parser.add_argument("--all",       action="store_true", help="Run full benchmark suite")
    args = parser.parse_args()

    results = []

    if args.all:
        # Full benchmark: sweep n_envs
        for robot in ["humanoid", "h1"]:
            for n in [1, 4, 8, 16, 32]:
                r = run_benchmark(robot, "walk", n, 1000)
                results.append(r)
    else:
        r = run_benchmark(args.robot, args.task, args.n_envs, args.steps, args.subproc)
        results.append(r)

    # Print table
    print(f"\n{'Robot':<12} {'Task':<8} {'N_envs':>8} {'Steps/sec':>12}")
    print("-" * 44)
    for r in results:
        print(f"{r['robot']:<12} {r['task']:<8} {r['n_envs']:>8} {r['steps_per_sec']:>12,.0f}")


if __name__ == "__main__":
    main()
