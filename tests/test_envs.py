"""
tests/test_envs.py
Environment smoke tests — verify envs can be created, stepped, and reset.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest


def make_humanoid_env(task="walk"):
    """Create MuJoCo built-in humanoid env via robot_cfg."""
    import mujoco
    from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
    from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
    from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
    from humanoid_rl.envs.mujoco.tasks.run_task import RunTask
    from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask
    from humanoid_rl.envs.base_env import BaseHumanoidEnv
    from humanoid_rl.rewards.locomotion import LocomotionRewards
    from humanoid_rl.rewards.stability  import StabilityRewards
    from humanoid_rl.rewards.energy     import EnergyRewards

    task_map = {"walk": WalkTask, "run": RunTask, "jump": JumpTask}
    task_obj = task_map.get(task, WalkTask)()
    task_cfg = task_obj.to_cfg()
    robot_cfg = get_humanoid_cfg({"min_torso_height": 0.4})
    xml_path = robot_cfg["mjcf_path_absolute"]

    class BuiltinEnv(MuJoCoHumanoidEnv):
        def __init__(self):
            BaseHumanoidEnv.__init__(self, robot_cfg, task_cfg, None, 5)
            self._width = 64; self._height = 64
            self._model = mujoco.MjModel.from_xml_path(xml_path)
            self._data  = mujoco.MjData(self._model)
            self._sim_dt  = self._model.opt.timestep
            self._ctrl_dt = self._sim_dt * 5
            from gymnasium import spaces
            obs_dim = self._compute_obs_dim()
            act_dim = self._model.nu
            self._observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            self._action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
            self._torque_limits = np.ones(act_dim) * robot_cfg.get("default_torque_limit", 100.0)
            rwd = task_cfg.get("rewards", {})
            self._loco_rewards   = LocomotionRewards(rwd.get("locomotion", {}))
            self._stab_rewards   = StabilityRewards(rwd.get("stability", {}))
            self._energy_rewards = EnergyRewards(rwd.get("energy", {}))
            self._max_steps    = task_cfg.get("max_steps", 1000)
            self._target_speed = task_cfg.get("target_speed", 1.0)
            self._renderer = None; self._viewer = None
            self._init_qpos = self._data.qpos.copy()
            self._init_qvel = self._data.qvel.copy()

    return BuiltinEnv()


class TestEnvironments:

    def test_reset(self):
        env = make_humanoid_env("walk")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape[0] == env.observation_space.shape[0]
        assert not np.any(np.isnan(obs))
        env.close()

    def test_step(self):
        env = make_humanoid_env("walk")
        obs, _ = env.reset(seed=0)
        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert obs2.shape == obs.shape
        env.close()

    def test_full_episode(self):
        """Run until done or 200 steps."""
        env = make_humanoid_env("walk")
        obs, _ = env.reset(seed=1)
        total_reward = 0.0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            if term or trunc:
                break
        assert np.isfinite(total_reward)
        env.close()

    @pytest.mark.parametrize("task", ["walk", "run", "jump"])
    def test_tasks(self, task):
        env = make_humanoid_env(task)
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] > 0
        env.close()


class TestVecEnv:

    def test_sync_vec_env(self):
        from humanoid_rl.utils.vec_env_wrapper import SyncVecEnv
        n_envs = 4
        envs = SyncVecEnv([lambda: make_humanoid_env("walk") for _ in range(n_envs)])
        obs, _ = envs.reset()
        assert obs.shape == (n_envs, envs.obs_dim)
        actions = np.random.uniform(-1, 1, (n_envs, envs.act_dim)).astype(np.float32)
        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        assert next_obs.shape == obs.shape
        assert rewards.shape == (n_envs,)
        envs.close()


class TestDataset:

    def test_write_read(self, tmp_path):
        from humanoid_rl.data.hdf5_dataset import HDF5Dataset
        obs_dim = 45; act_dim = 17; T = 50

        path = str(tmp_path / "test.hdf5")
        ds = HDF5Dataset(path, robot="humanoid", task="walk", mode="w")
        episode = {
            "observations": [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)],
            "actions":      [np.random.uniform(-1,1,act_dim).astype(np.float32) for _ in range(T)],
            "rewards":      np.random.randn(T).tolist(),
            "dones":        [False] * (T-1) + [True],
            "next_obs":     [np.random.randn(obs_dim).astype(np.float32) for _ in range(T)],
        }
        idx = ds.write_episode(episode)
        assert idx == 0
        ds.close()

        ds2 = HDF5Dataset(path, mode="r")
        ep = ds2.get_episode(0)
        assert ep["observations"].shape == (T, obs_dim)
        assert ep["actions"].shape == (T, act_dim)
        ds2.close()

    def test_cpp_rollout_buffer(self):
        """Test C++ RolloutBuffer if available."""
        try:
            import humanoid_rl_cpp as cpp
        except ImportError:
            pytest.skip("C++ extension not built")

        buf = cpp.RolloutBuffer(256, 45, 17)
        t = cpp.Transition()
        t.obs      = [0.1] * 45
        t.action   = [0.5] * 17
        t.reward   = 1.0
        t.done     = False
        t.next_obs = [0.2] * 45
        t.value    = 0.9
        t.log_prob = -1.2

        assert buf.push(t)
        assert len(buf) == 1
        data = buf.drain()
        assert data["obs"].shape   == (1, 45)
        assert data["action"].shape == (1, 17)
        assert len(buf) == 0
