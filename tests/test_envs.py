"""
tests/test_envs.py
Environment smoke tests — verify envs can be created, stepped, and reset.
Tests both generic humanoid (MuJoCoHumanoidEnv) and H1.
"""

import sys, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np
import pytest

# ─── Generic humanoid env tests ────────────────────────────────────────────

def make_gen_env(task="walk"):
    """Create generic MuJoCo humanoid env (FIXED: with PD control)."""
    from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
    from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
    from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
    from humanoid_rl.envs.mujoco.tasks.run_task import RunTask
    from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask
    from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask

    task_map = {"walk": WalkTask, "run": RunTask, "jump": JumpTask, "stand": StandTask}
    task_obj = task_map.get(task, WalkTask)()
    task_cfg = task_obj.to_cfg()
    robot_cfg = get_humanoid_cfg({"min_torso_height": 0.3, "reset_noise_scale": 0.01})
    return MuJoCoHumanoidEnv(
        robot_cfg=robot_cfg,
        task_cfg=task_cfg,
        frame_skip=5,
    )

def make_h1_env(task="walk"):
    """Create H1 env."""
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
    from humanoid_rl.envs.mujoco.tasks.run_task import RunTask
    from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask
    from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
    from humanoid_rl.envs.mujoco.tasks.stair_task import StairTask

    task_map = {"walk": WalkTask, "run": RunTask, "jump": JumpTask,
                "stand": StandTask, "stair": StairTask}
    task_obj = task_map.get(task, WalkTask)()
    task_cfg = task_obj.to_cfg()
    return H1Env(task_cfg=task_cfg, frame_skip=10, history_len=3)

class TestGenericHumanoid:
    def test_reset(self):
        env = make_gen_env("walk")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape[0] == env.observation_space.shape[0]
        assert not np.any(np.isnan(obs))
        env.close()

    def test_step(self):
        env = make_gen_env("walk")
        obs, _ = env.reset(seed=0)
        action = env.action_space.sample()
        obs2, reward, term, trunc, info = env.step(action)
        assert isinstance(reward, float)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert obs2.shape == obs.shape
        env.close()

    def test_full_episode(self):
        env = make_gen_env("walk")
        obs, _ = env.reset(seed=1)
        total = 0.0
        for _ in range(200):
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            total += reward
            if term or trunc:
                break
        assert np.isfinite(total)
        env.close()

    def test_pd_control_stand(self):
        """Zero action → PD control keeps humanoid upright."""
        from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
        from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
        from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
        robot_cfg = get_humanoid_cfg({"min_torso_height": 0.3})
        env = MuJoCoHumanoidEnv(robot_cfg=robot_cfg, task_cfg=StandTask().to_cfg(), frame_skip=5)
        env.reset(seed=0)
        for _ in range(100):
            _, r, _, _, _ = env.step(np.zeros(env.model.nu, dtype=np.float32))
        torso_z = env._get_body_pos(env._torso_body)[2]
        assert torso_z > 0.4, f"Robot fell with PD! torso_z={torso_z}"
        env.close()

    @pytest.mark.parametrize("task", ["walk", "run", "jump", "stand", "stair"])
    def test_tasks(self, task):
        env = make_gen_env(task)
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] > 0
        env.close()

class TestH1:
    def test_reset(self):
        env = make_h1_env("stand")
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert obs.shape[0] == env.observation_space.shape[0]
        assert not np.any(np.isnan(obs))
        env.close()

    def test_step(self):
        env = make_h1_env("stand")
        obs, _ = env.reset(seed=0)
        obs2, reward, term, trunc, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert obs2.shape == obs.shape
        env.close()

    def test_full_episode(self):
        env = make_h1_env("stand")
        obs, _ = env.reset(seed=1)
        for _ in range(200):
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                break
        assert np.isfinite(reward)
        env.close()

    @pytest.mark.parametrize("task", ["walk", "run", "jump", "stand", "stair"])
    def test_tasks(self, task):
        env = make_h1_env(task)
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] > 0
        env.close()

# ─── VecEnv ─────────────────────────────────────────────────────────────────

class TestVecEnv:
    def test_sync_vec_env(self):
        from humanoid_rl.utils.vec_env_wrapper import SyncVecEnv
        from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
        from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
        from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
        robot_cfg = get_humanoid_cfg({"min_torso_height": 0.3})
        task_cfg = WalkTask().to_cfg()
        def make():
            return MuJoCoHumanoidEnv(robot_cfg=robot_cfg, task_cfg=task_cfg, frame_skip=5)
        n_envs = 4
        envs = SyncVecEnv([make for _ in range(n_envs)])
        obs, _ = envs.reset()
        assert obs.shape == (n_envs, envs.obs_dim)
        actions = np.random.uniform(-1, 1, (n_envs, envs.act_dim)).astype(np.float32)
        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        assert next_obs.shape == obs.shape
        assert rewards.shape == (n_envs,)
        envs.close()

    def test_subproc_vec_env(self):
        from humanoid_rl.utils.vec_env_wrapper import SubprocVecEnv
        from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
        from humanoid_rl.robots.mujoco_humanoid import get_humanoid_cfg
        from humanoid_rl.envs.mujoco.humanoid_loco import MuJoCoHumanoidEnv
        robot_cfg = get_humanoid_cfg({"min_torso_height": 0.3})
        task_cfg = WalkTask().to_cfg()
        def make():
            return MuJoCoHumanoidEnv(robot_cfg=robot_cfg, task_cfg=task_cfg, frame_skip=5)
        n_envs = 2
        envs = SubprocVecEnv([make for _ in range(n_envs)])
        obs, _ = envs.reset()
        assert obs.shape == (n_envs, envs.obs_dim)
        actions = np.random.uniform(-1, 1, (n_envs, envs.act_dim)).astype(np.float32)
        next_obs, rewards, terms, truncs, _ = envs.step(actions)
        assert next_obs.shape == obs.shape
        envs.close()

# ─── Dataset ────────────────────────────────────────────────────────────────

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
        try:
            import humanoid_rl_cpp as cpp
        except ImportError:
            pytest.skip("C++ extension not built")
        buf = cpp.RolloutBuffer(256, 45, 17)
        t = cpp.Transition()
        t.obs = [0.1] * 45
        t.action = [0.5] * 17
        t.reward = 1.0
        t.done = False
        t.next_obs = [0.2] * 45
        t.value = 0.9
        t.log_prob = -1.2
        assert buf.push(t)
        assert len(buf) == 1
        data = buf.drain()
        assert data["obs"].shape == (1, 45)
        assert data["action"].shape == (1, 17)
