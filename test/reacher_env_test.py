import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as th

from aegis_gym.envs import ReacherEnv

# the TCP above the object by the target offset, with the object's orientation
_TARGET_TCP_POSE = [0.4, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0]


def _yawed_quat(deg: float) -> list[float]:
    half = math.radians(deg) / 2
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def _reacher_env(tcp_poses: list[list[float]]) -> ReacherEnv:
    """The env with the simulation stubbed out: the object at the origin, the TCPs given."""
    num_envs = len(tcp_poses)
    env = object.__new__(ReacherEnv)
    env._scene = None  # checked by `BaseEnv.__del__()`
    env.device, env.num_envs = th.device("cpu"), num_envs
    obj_pose = th.tensor([[0.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
    env.manipulator = MagicMock(get_tcp_pose=lambda: th.tensor(tcp_poses))
    env.object = MagicMock(get_pose=lambda: obj_pose)
    env.success_pos_thresh, env.success_rot_thresh = 0.01, math.radians(5.0)
    env._cfg_env = SimpleNamespace(episode_length_s=5.0)
    env.episode_length_buf = th.zeros(num_envs)
    env.goal_pose = th.zeros(num_envs, 7)
    env.success_buf = th.zeros(num_envs, dtype=th.bool)
    env.episode_sums = {}
    env.extras = {}
    env._is_domain_randomized = lambda: False
    env._get_random_object_pose = lambda num_reset: obj_pose[:num_reset]
    return env


def test_reacher_tcp_to_target_errors():
    env = _reacher_env(
        [
            _TARGET_TCP_POSE,
            [0.4, 0.02, 0.08, *_yawed_quat(10.0)],
        ]
    )

    pos_err, rot_err = env._tcp_to_target_errors()

    assert pos_err.tolist() == pytest.approx([0.0, 0.02], abs=1e-6)
    assert th.rad2deg(rot_err).tolist() == pytest.approx([0.0, 10.0], abs=1e-3)


def test_reacher_reports_the_success_of_the_episodes_last_step():
    # within, beyond the position and beyond the rotation thresholds
    env = _reacher_env(
        [
            [0.405, 0.0, 0.08, *_yawed_quat(3.0)],
            [0.42, 0.0, 0.08, 1.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.08, *_yawed_quat(8.0)],
        ]
    )
    env.tcp_pos_err, env.tcp_rot_err = env._tcp_to_target_errors()
    env.success_buf = (env.tcp_pos_err < env.success_pos_thresh) & (
        env.tcp_rot_err < env.success_rot_thresh
    )
    assert env.success_buf.tolist() == [True, False, False]

    env.reset_idx(th.tensor([0, 1]))

    episode = env.extras["episode"]
    assert episode["success_rate"] == pytest.approx(0.5)
    assert episode["final_tcp_dist_cm"] == pytest.approx((0.5 + 2.0) / 2, abs=1e-4)
    assert episode["final_rot_err_deg"] == pytest.approx(3.0 / 2, abs=1e-3)
    # cleared for the next episodes
    assert env.success_buf.tolist() == [False, False, False]
