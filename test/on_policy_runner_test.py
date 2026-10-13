from types import SimpleNamespace
from unittest.mock import MagicMock

import torch as th

import aegis_gym.config  # noqa: F401
from aegis_gym.runners import OnPolicyRunner


class _FakeRslRunner:
    """Mimics the rsl_rl runner: `learn()` accumulates the running episode statistics."""

    def __init__(self, num_envs: int, rnd: bool = False):
        self.current_learning_iteration = 0
        self.alg = SimpleNamespace(policy=MagicMock())
        self.logger = SimpleNamespace(
            cur_reward_sum=th.zeros(num_envs),
            cur_episode_length=th.zeros(num_envs),
        )
        if rnd:
            self.logger.cur_ereward_sum = th.zeros(num_envs)
            self.logger.cur_ireward_sum = th.zeros(num_envs)
        self.sums_at_learn_start: list[float] = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool) -> None:
        self.sums_at_learn_start.append(self.logger.cur_reward_sum.sum().item())
        start = self.current_learning_iteration
        for it in range(start, start + num_learning_iterations):
            for buffer in vars(self.logger).values():
                buffer += 1.0
            self.current_learning_iteration = it


def _make_runner(rsl_runner: _FakeRslRunner) -> OnPolicyRunner:
    runner = object.__new__(OnPolicyRunner)
    runner.runner = rsl_runner
    return runner


def test_learn_in_chunks_iterations():
    rsl_runner = _FakeRslRunner(num_envs=4)
    chunk_ends: list[int] = []
    _make_runner(rsl_runner).learn_in_chunks(
        num_learning_iterations=25, chunk_iterations=10, on_chunk_end=chunk_ends.append
    )
    assert chunk_ends == [9, 19]
    assert rsl_runner.current_learning_iteration == 24


def test_learn_in_chunks_discards_interrupted_episodes():
    rsl_runner = _FakeRslRunner(num_envs=4, rnd=True)
    sums_at_chunk_end: list[float] = []

    def on_chunk_end(_iteration: int) -> None:
        sums_at_chunk_end.append(rsl_runner.logger.cur_reward_sum.sum().item())

    _make_runner(rsl_runner).learn_in_chunks(
        num_learning_iterations=30, chunk_iterations=10, on_chunk_end=on_chunk_end
    )
    # The statistics accumulated before the preview are dropped, not carried into the next chunk
    assert all(s > 0 for s in sums_at_chunk_end)
    assert rsl_runner.sums_at_learn_start == [0.0, 0.0, 0.0]
    assert rsl_runner.alg.policy.reset.call_count == 2


def test_discard_interrupted_episodes_without_rnd():
    rsl_runner = _FakeRslRunner(num_envs=2)
    rsl_runner.logger.cur_reward_sum += 3.0
    rsl_runner.logger.cur_episode_length += 5.0
    _make_runner(rsl_runner)._discard_interrupted_episodes()
    assert not rsl_runner.logger.cur_reward_sum.any()
    assert not rsl_runner.logger.cur_episode_length.any()
