"""
tests/test_training_correctness.py
Regression tests for RL-pipeline bugs that silently ruin policies.
"""
import os, sys
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import numpy as np

from humanoid_rl.algorithms.ppo.rollout_buffer import PPORolloutBuffer


def _buffer(rewards, dones, values=None, gamma=1.0, lam=1.0):
    T = len(rewards)
    b = PPORolloutBuffer(T, 1, 2, 1, gamma=gamma, gae_lambda=lam)
    values = values or [0.0] * T
    for r, d, v in zip(rewards, dones, values):
        b.add(np.zeros((1, 2)), np.zeros((1, 1)), np.array([r]), np.array([d]),
              np.array([v]), np.array([0.0]))
    return b


def test_gae_does_not_leak_across_episode_end():
    # Episode ends right after step 1 (done[1]=1). Steps 2,3 belong to a new episode.
    b = _buffer([0, 10, 5, 5], [0, 1, 0, 0])
    b.compute_returns_and_advantages(np.array([0.0]), np.array([0.0]))
    adv = b.advantages[:, 0]
    assert adv[1] == 10        # terminal step: reward only, no bootstrap
    assert adv[0] == 10        # sees the terminal reward
    assert adv[2] == 10        # new episode: 5 + 5, none of episode 1


def test_gae_last_step_done_has_no_bootstrap():
    b = _buffer([1, 1], [0, 1])
    b.compute_returns_and_advantages(np.array([100.0]), np.array([1.0]))
    assert b.advantages[1, 0] == 1.0


def test_gait_foot_clearance_zero_when_standing():
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    from humanoid_rl.envs.mujoco.tasks.walk_task import WalkTask
    env = H1Env(WalkTask().to_cfg(), render_mode=None)
    env.reset(seed=0)
    for _ in range(8):
        env.step(np.zeros(19))
    st = env._build_state_dict(np.zeros(19))
    assert st["lfoot_z"] < 0.02 and st["rfoot_z"] < 0.02


def test_cpg_reference_reachable_by_pd_target():
    from humanoid_rl.envs.mujoco.h1_env import H1_DEFAULT_QPOS, H1_ACTION_SCALE_VEC
    from humanoid_rl.rewards.gait import GaitRewards
    g = GaitRewards({}, control_dt=0.02, default_q=H1_DEFAULT_QPOS)
    legs = [1, 2, 3, 4, 6, 7, 8, 9]
    for ph in np.linspace(0, 1, 50, endpoint=False):
        g._phase = ph
        q = g._cpg_reference()
        assert np.all(np.abs(q[legs] - H1_DEFAULT_QPOS[legs]) <= H1_ACTION_SCALE_VEC[legs] + 1e-6)


def test_default_pose_foot_below_hip():
    """Default pose must keep CoM roughly over the feet (not behind the heel)."""
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
    env = H1Env(StandTask().to_cfg(), render_mode=None)
    env.reset(seed=0)
    com_rel, _ = env._compute_com_state()
    assert abs(com_rel[0]) < 0.07


def test_task_terrain_is_used():
    from humanoid_rl.envs.mujoco.h1_env import H1Env
    from humanoid_rl.envs.mujoco.tasks.jump_task import JumpTask
    env = H1Env(JumpTask().to_cfg(), render_mode=None)
    assert env._terrain_type == "hurdle"


def test_run_forward_reward_has_gradient_from_standstill():
    from humanoid_rl.rewards.locomotion import LocomotionRewards
    lr = LocomotionRewards({"forward_vel_weight": 3.0, "lateral_vel_penalty": 0.0})
    r0 = lr.compute({"root_vel": np.zeros(3), "target_speed": 3.0, "torso_height": 1.0})
    assert r0 > 0.1


def test_height_gain_cannot_be_farmed_by_bobbing():
    from humanoid_rl.rewards.locomotion import LocomotionRewards
    lr = LocomotionRewards({"height_gain_weight": 1.0, "forward_vel_weight": 0.0,
                            "lateral_vel_penalty": 0.0})
    tot = 0.0
    for h in [1.0, 1.1, 1.0, 1.1, 1.0, 1.1, 1.0, 1.1]:
        tot += lr.compute({"root_vel": np.zeros(3), "target_speed": 0.0, "torso_height": h})
    assert tot < 0.11


def test_ankle_has_enough_balance_authority():
    """Ankle PD authority (kp*scale) must reach >=30 Nm; 12 Nm made standing impossible."""
    from humanoid_rl.envs.mujoco.h1_env import H1Env, H1_KP
    from humanoid_rl.envs.mujoco.tasks.stand_task import StandTask
    env = H1Env(StandTask().to_cfg(), render_mode=None)
    auth = H1_KP * env._pd.action_scale_vec
    assert auth[4] >= 30 and auth[9] >= 30
