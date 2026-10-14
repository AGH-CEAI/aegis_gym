from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as th
from clearml.utilities.proxy_object import ProxyDictPreWrite

from aegis_gym.envs import PushTEnv, ReacherEnv

NUM_ENVS = 3
MAX_EPISODE_LENGTH = 5


class _Scene:
    def pre_step(self) -> None: ...

    def step(self) -> None: ...

    def shutdown(self) -> None: ...  # called by `BaseEnv.__del__()`


def _stepping_env(success: list[bool], off_table: list[bool]) -> PushTEnv:
    """
    The env with the simulation stubbed out: the tee's success and off table states are given,
    and `reset_idx()` only restarts the episode.
    """
    env = object.__new__(PushTEnv)
    env._scene = _Scene()
    env.device = th.device("cpu")
    env.num_envs = NUM_ENVS
    env.max_episode_length = MAX_EPISODE_LENGTH
    env.episode_length_buf = th.zeros(NUM_ENVS, dtype=th.long)
    env.success_thresh = 0.9
    env.terminate_on_success = True
    env.reward_functions, env.reward_scales, env.episode_sums = {}, {}, {}
    env.extras = {"observations": {}}
    env.manipulator = SimpleNamespace(ctrl_apply_vel_action=lambda *a, **kw: None)
    env._planar_action_to_tcp_twist = lambda actions: actions
    env._tee_projection_to_goal_intersection = lambda: th.tensor(success).float()
    env._tee_to_goal_pose_distance = lambda: th.zeros(NUM_ENVS)
    env._tcp_to_tee_footprint_distance = lambda: th.zeros(NUM_ENVS)
    env._is_tee_off_table = lambda: th.tensor(off_table)
    env.get_observations = lambda: None
    env.reset_calls = []

    def reset_idx(envs_idx: th.Tensor) -> None:
        env.reset_calls.append(envs_idx.tolist())
        env.episode_length_buf[envs_idx] = 0
        # as the env's reset, clears the buffers in place
        env.success_buf[envs_idx] = False
        env.tee_off_table_buf[envs_idx] = False
        env._is_tee_off_table = lambda: th.zeros(NUM_ENVS, dtype=th.bool)

    env.reset_idx = reset_idx
    return env


def test_push_t_step_reports_the_terminated_episodes():
    # env 0: times out, env 1: tee off the table, env 2: running
    env = _stepping_env(success=[False, False, False], off_table=[False, True, False])
    env.episode_length_buf[0] = MAX_EPISODE_LENGTH

    _, _, dones, extras = env._step(th.zeros(NUM_ENVS, 2))

    # The dones and time outs are of the ended episodes, not of the restarted ones
    assert dones.tolist() == [True, True, False]
    assert extras["time_outs"].tolist() == [1.0, 0.0, 0.0]
    assert extras["tee_off_table"].tolist() == [False, True, False]
    assert env.reset_calls == [[0, 1]]


def test_push_t_step_resets_a_success_once():
    env = _stepping_env(success=[True, False, False], off_table=[False, False, False])

    _, _, dones, extras = env._step(th.zeros(NUM_ENVS, 2))

    assert dones.tolist() == [True, False, False]
    # The flag of the ended episode, though the reset cleared the env's buffer
    assert extras["success"].tolist() == [True, False, False]
    assert env.reset_calls == [[0]]


@pytest.mark.parametrize("env_cls", [PushTEnv, ReacherEnv])
@pytest.mark.parametrize("read_only", [False, True])
def test_reward_scales_are_scaled_copies(env_cls: type, read_only: bool):
    given = {"a": 10.0, "b": 1.0}
    # The connected config under the ClearML agent ignores the writes
    config_scales = (
        ProxyDictPreWrite(False, False, **given) if read_only else dict(given)
    )
    env = object.__new__(env_cls)
    env._scene = None  # checked by `BaseEnv.__del__()`
    env.device, env.num_envs = th.device("cpu"), NUM_ENVS
    env.ctrl_dt, env.sim_substeps = 0.004, 10
    env.reward_scales = config_scales
    env._reward_a = env._reward_b = lambda: th.zeros(NUM_ENVS)

    env._init_reward_functions()

    assert env.reward_scales == pytest.approx({"a": 0.4, "b": 0.04})
    assert dict(config_scales) == given


def test_push_t_reset_reports_the_ended_episodes_metrics():
    env = object.__new__(PushTEnv)
    env._scene = _Scene()
    env.device, env.num_envs = th.device("cpu"), NUM_ENVS
    env._cfg_env = SimpleNamespace(episode_length_s=30.0)
    env.episode_length_buf = th.zeros(NUM_ENVS, dtype=th.long)
    env.manipulator = MagicMock()
    env.object = MagicMock()
    env.tcp_start_pose = th.zeros(NUM_ENVS, 7)
    env.goal_rand_enabled = False
    env._is_domain_randomized = lambda: False
    env._get_random_tee_pose = lambda num_reset, seeds: th.zeros(num_reset, 7)
    env._tee_to_goal_pose_distance = lambda: th.zeros(NUM_ENVS)
    env._tcp_to_tee_footprint_distance = lambda: th.zeros(NUM_ENVS)
    env.last_pose_dist, env.last_tcp_dist = th.zeros(NUM_ENVS), th.zeros(NUM_ENVS)
    env.episode_sums, env.extras = {}, {}
    # the last step of the ended episodes (envs 0 and 1)
    env.success_buf = th.tensor([True, False, False])
    env.tee_off_table_buf = th.tensor([False, True, False])
    env.intersection_ratio = th.tensor([0.95, 0.15, 0.5])
    env.pose_dist = th.tensor([0.01, 0.2, 0.1])

    env.reset_idx(th.tensor([0, 1]))

    assert env.extras["episode"] == pytest.approx(
        {
            "success_rate": 0.5,
            "tee_off_table_rate": 0.5,
            "final_coverage": 0.55,
            "final_pose_dist_cm": 10.5,
        }
    )
