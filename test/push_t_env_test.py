import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch as th
from clearml.utilities.proxy_object import ProxyDictPreWrite

from aegis_gym.envs import BaseEnv, PushTEnv, ReacherEnv
from aegis_gym.envs.base_env import Modality

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
        env.extras["episode"] = {"success_rate": 1.0}
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


def test_episode_metrics_are_of_the_resets_of_the_step_only():
    env = _stepping_env(success=[True, False, False], off_table=[False, False, False])

    *_, extras = env.step(th.zeros(NUM_ENVS, 2))
    assert extras["episode"] == {"success_rate": 1.0}

    # no reset on the next step, the rsl_rl logger would count the earlier metrics again
    env._tee_projection_to_goal_intersection = lambda: th.zeros(NUM_ENVS)
    *_, extras = env.step(th.zeros(NUM_ENVS, 2))
    assert env.reset_calls == [[0]]
    assert "episode" not in extras


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

    # per ended episode, the rsl_rl logger averages them over all the episodes of the iteration
    expected = {
        "success_rate": [1.0, 0.0],
        "tee_off_table_rate": [0.0, 1.0],
        "final_coverage": [0.95, 0.15],
        "final_pose_dist_cm": [1.0, 20.0],
    }
    episode = env.extras["episode"]
    assert episode.keys() == expected.keys()
    for key, values in expected.items():
        assert episode[key].tolist() == pytest.approx(values), key


def test_push_t_randomized_goals_stay_on_the_table():
    num = 2000
    env = object.__new__(PushTEnv)
    env._scene = _Scene()
    env.device = th.device("cpu")
    env.goal_offset_xy, env.goal_z_rot = [0.47, 0.0], 0.0
    env.goal_rand_x, env.goal_rand_y = [-0.05, 0.05], [-0.1, 0.1]
    env.goal_rand_z_rot = [-th.pi, th.pi]
    env.goal_table_margin = 0.01
    env.table_x, env.table_y = (0.32, 0.87), (-0.42, 0.42)
    env.goal_yaw, env.goal_pose = th.zeros(num), th.zeros(num, 7)
    env._build_goal_transform = lambda envs_idx: None
    env.goal_marker = MagicMock()

    env._randomize_goal(envs_idx=th.arange(num), seeds=list(range(num)))

    extent_min, extent_max = env._tee_footprint_extent(env.goal_yaw)
    footprint_min = env.goal_pose[:, :2] + extent_min
    footprint_max = env.goal_pose[:, :2] + extent_max
    assert footprint_min[:, 0].min() >= 0.32 + 0.01 - 1e-6
    assert footprint_max[:, 0].max() <= 0.87 - 0.01 + 1e-6
    assert footprint_min[:, 1].min() >= -0.42 + 0.01 - 1e-6
    assert footprint_max[:, 1].max() <= 0.42 - 0.01 + 1e-6
    # a subset of the configured ranges, so the teachers trained on them stay in distribution
    assert env.goal_pose[:, 0].min() >= 0.42 and env.goal_pose[:, 0].max() <= 0.52
    assert env.goal_pose[:, 1].abs().max() <= 0.1


def _state_obs_env(env_cls: type = PushTEnv) -> BaseEnv:
    """The env with its state terms on stubbed TCP, tee (90 deg about z) and goal poses."""
    env = object.__new__(env_cls)
    cfg = SimpleNamespace(
        env_cfg=SimpleNamespace(num_envs=NUM_ENVS),
        dr_cfg=None,
        get_device=lambda: th.device("cpu"),
    )
    BaseEnv.__init__(env, scene=_Scene(), cfg=cfg)
    tcp = th.zeros(NUM_ENVS, 7)
    tcp[:, :3] = th.tensor([0.5, 0.1, 0.2])
    reference = tcp.clone()
    reference[:, :2] += th.tensor([0.01, -0.02])
    tee = th.zeros(NUM_ENVS, 7)
    tee[:, :2] = th.tensor([0.4, 0.1])
    tee[:, 3] = math.cos(math.pi / 4)  # wxyz
    tee[:, 6] = math.sin(math.pi / 4)
    env._observation_fns = {
        Modality.TCP_POSE: lambda: tcp,
        Modality.OBJECT_POSE: lambda: tee,
    }
    env.manipulator = SimpleNamespace(get_tcp_reference=lambda: reference)
    env.goal_pose = th.zeros(NUM_ENVS, 7)
    env.goal_pose[:, 0] = 0.47
    env.goal_yaw = th.zeros(NUM_ENVS)
    env.get_observations = lambda: {"policy": th.full((NUM_ENVS, 3), 7.0)}
    return env


def test_state_obs_concatenates_the_named_terms():
    env = _state_obs_env()

    state = env.get_state_obs(["tcp_pose", "reference_offset", "teacher_obs"])

    assert state.shape == (NUM_ENVS, 7 + 2 + 3)
    assert state[0, :3].tolist() == pytest.approx([0.5, 0.1, 0.2])
    assert state[0, 7:9].tolist() == pytest.approx([0.01, -0.02])
    assert state[0, 9:].tolist() == [7.0, 7.0, 7.0]


def test_push_t_tee_and_goal_state_obs_are_planar():
    state = _state_obs_env().get_state_obs(["tee_pose", "goal_pose"])

    assert state[0].tolist() == pytest.approx(
        [0.4, 0.1, 1.0, 0.0, 0.47, 0.0, 0.0, 1.0], abs=1e-6
    )


def test_state_obs_rejects_the_unknown_names_and_allows_none():
    env = _state_obs_env(ReacherEnv)

    assert env.available_state_obs == {"tcp_pose", "teacher_obs"}
    with pytest.raises(ValueError, match="Unknown state_obs.*ReacherEnv"):
        env.get_state_obs(["tcp_pose", "tee_pose"])
    assert env.get_state_obs([]).shape == (NUM_ENVS, 0)


class _UnseededReacherEnv(ReacherEnv):
    _reset_idx_seeded = BaseEnv._reset_idx_seeded


@pytest.mark.parametrize(
    ("env_cls", "supported"),
    [(PushTEnv, True), (ReacherEnv, True), (_UnseededReacherEnv, False)],
)
def test_envs_report_the_seeded_reset_support(env_cls: type, supported: bool):
    env = object.__new__(env_cls)
    env._scene = _Scene()

    assert env.supports_seeded_reset is supported


def test_push_t_contact_force_is_the_world_xy_force_scaled():
    env = _state_obs_env()
    tcp = th.zeros(NUM_ENVS, 7)
    tcp[:, 3] = math.cos(math.pi / 4)  # wxyz, the sensor frame 90 deg about z
    tcp[:, 6] = math.sin(math.pi / 4)
    wrench = th.zeros(NUM_ENVS, 6)
    wrench[:, :3] = th.tensor([10.0, 0.0, 3.0])
    env._observation_fns[Modality.TCP_POSE] = lambda: tcp
    env.manipulator.get_ft_wrench = lambda: wrench
    env.force_obs_scale = 0.002

    state = env.get_state_obs(["contact_force"])

    assert state.shape == (NUM_ENVS, 2)
    assert state[0].tolist() == pytest.approx([0.0, 0.02], abs=1e-6)
