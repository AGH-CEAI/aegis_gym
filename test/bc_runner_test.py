import math
from types import SimpleNamespace

import pytest
import torch as th

import aegis_gym.config  # noqa: F401
from aegis_gym.runners.bc_runner import (
    BehaviorCloningRunner,
    _mean_episode_extras,
    _pose_errors,
)

NUM_ENVS = 2


class _FakeLogger:
    """Records the env steps passed to the rsl_rl logger."""

    def __init__(self):
        self.steps: list[tuple[list[bool], dict]] = []
        self.cur_reward_sum = th.ones(NUM_ENVS)
        self.cur_episode_length = th.ones(NUM_ENVS)

    def process_env_step(
        self, rewards: th.Tensor, dones: th.Tensor, extras: dict
    ) -> None:
        # a copy, the env clears the key on the next step
        self.steps.append((dones.tolist(), dict(extras)))


class _FakeEnv:
    """Ends the episode of env 0 on the 2nd step, with its `extras["episode"]` of that step only."""

    def __init__(self):
        self.num_steps = 0
        self.extras = {}
        tcp_pose = th.zeros(NUM_ENVS, 7)
        self.manipulator = SimpleNamespace(get_tcp_pose=lambda: tcp_pose)
        self.object = SimpleNamespace(get_pose=lambda: th.zeros(NUM_ENVS, 7))

    def get_observations(self) -> th.Tensor:
        return th.zeros(NUM_ENVS, 1)

    def get_modality_observations(self, modalities) -> dict:
        return {m: th.zeros(NUM_ENVS, 3, 4, 4) for m in modalities}

    def step(self, actions: th.Tensor):
        self.num_steps += 1
        self.extras.pop("episode", None)  # as `BaseEnv.step()`
        dones = th.tensor([self.num_steps == 2, False])
        if self.num_steps == 2:
            self.extras["episode"] = {"success_rate": 1.0}
        return self.get_observations(), th.zeros(NUM_ENVS), dones, self.extras


def _collecting_runner(teacher_mixing: bool) -> BehaviorCloningRunner:
    runner = object.__new__(BehaviorCloningRunner)
    runner.env = _FakeEnv()
    runner.logger = _FakeLogger()
    runner._num_steps_per_env = 3
    runner._use_teacher_mixing = teacher_mixing
    # env 0: the student is close to the teacher, env 1: far (the teacher takes over)
    teacher_actions = th.tensor([[0.5, 0.0], [2.0, 0.0]])
    runner._teacher = lambda obs: teacher_actions
    runner._policy = lambda rgb, pose: th.tensor([[0.5, 0.0], [0.0, 0.0]])
    runner._buffer = SimpleNamespace(add=lambda *args: None)
    return runner


def test_collection_reports_the_student_vs_teacher_statistics():
    stats = _collecting_runner(teacher_mixing=True)._collect_with_rl_teacher()

    assert stats["teacher_intervention_rate"] == pytest.approx(0.5)
    # (0 + 2^2) / 4 action entries
    assert stats["action_error_onpolicy"] == pytest.approx(1.0)
    # the executed (clamped) norms: student 0.5 / 2, teacher (0.5 + 1) / 2
    assert stats["action_norm_ratio"] == pytest.approx(0.5 / 1.5)
    assert stats["teacher_action_saturation"] == pytest.approx(0.25)


def test_collection_without_mixing_has_no_intervention_rate():
    stats = _collecting_runner(teacher_mixing=False)._collect_with_rl_teacher()

    assert "teacher_intervention_rate" not in stats


def test_collection_passes_the_episode_metrics_of_the_ended_episodes():
    runner = _collecting_runner(teacher_mixing=False)

    runner._collect_with_rl_teacher()

    assert [extras for _, extras in runner.logger.steps] == [
        {},
        {"episode": {"success_rate": 1.0}},
        {},
    ]


def test_mean_episode_extras():
    ep_extras = [
        {"success_rate": 1.0, "rew_a": th.tensor(2.0)},
        {"success_rate": 0.0, "rew_a": th.tensor([4.0, 6.0])},
    ]

    assert _mean_episode_extras(ep_extras) == pytest.approx(
        {"success_rate": 0.5, "rew_a": 3.5}
    )


def test_pose_errors_in_physical_units():
    half_yaw = math.radians(90.0) / 2
    target = th.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    # 2 heads: 1 cm off and rotated by 90 deg about z (sign flipped quaternion: the same rotation)
    pred = th.tensor(
        [
            [[0.01, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, -math.cos(half_yaw), 0.0, 0.0, -math.sin(half_yaw)]],
        ]
    )

    pos_err, rot_err = _pose_errors(pred, target)

    assert pos_err == pytest.approx(0.005)
    assert math.degrees(rot_err) == pytest.approx(45.0, abs=1e-3)


def _best_model_runner(saved: list[int]) -> BehaviorCloningRunner:
    runner = object.__new__(BehaviorCloningRunner)
    runner.cfg_train = SimpleNamespace(best_model_skip_iters=0)
    runner._best_model_score = (-math.inf, -math.inf)
    runner._best_model_iter = -1
    runner._log_dir = "/nonexistent"
    runner.save = lambda path: None
    runner.logger = SimpleNamespace(save_model=lambda path, it: saved.append(it))
    return runner


def test_best_model_is_selected_by_the_preview_success_then_return():
    saved: list[int] = []
    runner = _best_model_runner(saved)

    runner._update_best_model(0, score=(-1.0, 50.0))  # without the preview: the reward
    runner._update_best_model(1, score=(0.2, 10.0))  # any preview success beats it
    runner._update_best_model(2, score=(0.2, 5.0))  # same success, lower return
    runner._update_best_model(3, score=(0.4, 1.0))

    assert saved == [1, 2, 4]
    assert runner._best_model_score == (0.4, 1.0)


def test_discard_interrupted_episodes():
    runner = object.__new__(BehaviorCloningRunner)
    runner.logger = _FakeLogger()

    runner._discard_interrupted_episodes()

    assert not runner.logger.cur_reward_sum.any()
    assert not runner.logger.cur_episode_length.any()
