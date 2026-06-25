"""
humanoid_rl/utils/vec_env_wrapper.py
Synchronous vectorized environment wrapper.

Runs N environments in parallel using Python multiprocessing.
Uses subprocess workers to avoid MuJoCo's OpenGL context conflicts.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np
from typing import Any, Callable, List, Tuple


# ─── Worker process ────────────────────────────────────────────────────────────

def _worker(env_fn: Callable, conn: mp.Connection, parent_conn: mp.Connection) -> None:
    """Worker process: receives commands, steps env, sends results."""
    parent_conn.close()
    env = env_fn()
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, reward, term, trunc, info = env.step(data)
                conn.send((obs, reward, term, trunc, info))
            elif cmd == "reset":
                obs, info = env.reset(**data)
                conn.send((obs, info))
            elif cmd == "close":
                env.close()
                break
            elif cmd == "get_spaces":
                conn.send((env.observation_space, env.action_space))
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    finally:
        env.close()


# ─── SyncVecEnv (in-process, no overhead) ─────────────────────────────────────

class SyncVecEnv:
    """
    Synchronous vectorized env — runs envs sequentially in the main process.
    Simpler than subprocess, avoids IPC overhead for fast envs.

    For MuJoCo with many envs (>32), consider SubprocVecEnv.
    """

    def __init__(self, env_fns: List[Callable]):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

        self.observation_space = self.envs[0].observation_space
        self.action_space      = self.envs[0].action_space

        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def reset(self, **kwargs) -> Tuple[np.ndarray, List[dict]]:
        results = [env.reset(**kwargs) for env in self.envs]
        obs_list, info_list = zip(*results)
        return np.stack(obs_list, axis=0), list(info_list)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        results = [env.step(actions[i]) for i, env in enumerate(self.envs)]
        obs_l, rew_l, term_l, trunc_l, info_l = zip(*results)
        return (
            np.stack(obs_l,  axis=0),
            np.array(rew_l,  dtype=np.float32),
            np.array(term_l, dtype=bool),
            np.array(trunc_l, dtype=bool),
            list(info_l),
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()


# ─── SubprocVecEnv (subprocess, true parallel) ────────────────────────────────

class SubprocVecEnv:
    """
    Subprocess vectorized env — each env runs in a separate process.
    Uses multiprocessing Pipes for communication.
    Best for heavy envs (>16 envs on many-core machines).
    """

    def __init__(self, env_fns: List[Callable]):
        self.n_envs = len(env_fns)
        self.waiting = False

        self.parent_conns, self.worker_conns = zip(*[mp.Pipe() for _ in env_fns])
        self.processes = []
        for fn, wconn, pconn in zip(env_fns, self.worker_conns, self.parent_conns):
            p = mp.Process(target=_worker, args=(fn, wconn, pconn), daemon=True)
            p.start()
            self.processes.append(p)
            wconn.close()  # close child-side in parent process

        # Get spaces from first worker
        self.parent_conns[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.parent_conns[0].recv()

        obs_dim = self.observation_space.shape[0]
        act_dim = self.action_space.shape[0]
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def reset(self, **kwargs) -> Tuple[np.ndarray, List[dict]]:
        for conn in self.parent_conns:
            conn.send(("reset", kwargs))
        results = [conn.recv() for conn in self.parent_conns]
        obs_list, info_list = zip(*results)
        return np.stack(obs_list, axis=0), list(info_list)

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        for i, conn in enumerate(self.parent_conns):
            conn.send(("step", actions[i]))
        results = [conn.recv() for conn in self.parent_conns]
        obs_l, rew_l, term_l, trunc_l, info_l = zip(*results)
        return (
            np.stack(obs_l,  axis=0),
            np.array(rew_l,  dtype=np.float32),
            np.array(term_l, dtype=bool),
            np.array(trunc_l, dtype=bool),
            list(info_l),
        )

    def close(self) -> None:
        for conn in self.parent_conns:
            conn.send(("close", None))
        for p in self.processes:
            p.join()
