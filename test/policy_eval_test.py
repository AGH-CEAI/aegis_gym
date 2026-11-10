import csv
from contextlib import contextmanager

import numpy as np
import pytest
import torch as th

from aegis_gym.runners.policy_eval import PolicyEvaluator, wilson_interval

NUM_ENVS = 4


class _SeededEnv:
    """
    A vectorized env whose seeded episode of the seed `s` lasts `s % 5 + 1` steps and succeeds for
    the even seeds; the unseeded ones (the auto resets) run long. Records the domain
    randomizations drawn by the evaluation and whether it ran in the nominal domain.
    """

    device = th.device("cpu")
    num_envs = NUM_ENVS
    max_episode_length = 20

    def __init__(self, dr_enabled: bool = True):
        self._dr_enabled = dr_enabled
        self.seed = [-1] * NUM_ENVS
        self.t = th.zeros(NUM_ENVS, dtype=th.long)
        self.in_nominal_domain = False
        self.nominal_steps = 0
        self.randomizations: list[float] = []
        self.observed_noise: list[float] = []
        self.extras: dict = {}

    def get_policy_dt(self) -> float:
        return 0.1

    def is_domain_randomization_enabled(self) -> bool:
        return self._dr_enabled

    def apply_domain_randomization(self, envs_idx: th.Tensor) -> None:
        self.randomizations.append(float(th.rand(1)))

    @contextmanager
    def nominal_domain(self):
        self.in_nominal_domain = True
        try:
            yield
        finally:
            self.in_nominal_domain = False

    def _reset_idx_seeded(self, envs_idx, seeds) -> None: ...

    def reset(self):
        for i in range(NUM_ENVS):
            self.seed[i] = -1
        self.t.zero_()

    def reset_seeded(self, envs_idx: th.Tensor, seeds) -> th.Tensor:
        for i, s in zip(envs_idx.tolist(), seeds, strict=True):
            self.seed[i] = s
            self.t[i] = 0
        return self._obs()

    def _obs(self) -> th.Tensor:
        return th.tensor(self.seed, dtype=th.float32).unsqueeze(-1)

    def _length(self, i: int) -> int:
        return self.seed[i] % 5 + 1 if self.seed[i] >= 0 else 1000

    def step(self, actions: th.Tensor):
        self.nominal_steps += int(self.in_nominal_domain)
        self.observed_noise.append(float(th.rand(1)))
        self.t += 1
        dones = th.tensor([int(self.t[i]) >= self._length(i) for i in range(NUM_ENVS)])
        success = th.tensor(
            [bool(dones[i]) and self.seed[i] % 2 == 0 for i in range(NUM_ENVS)]
        )
        reset_envs = th.nonzero(dones).flatten().tolist()
        extras = {
            "success": success,
            "time_outs": th.zeros(NUM_ENVS),
            "episode": {
                "final_coverage": th.tensor(
                    [self.seed[i] / 1000 for i in reset_envs], dtype=th.float32
                ),
                "success_rate": success[dones].float(),
                "rew_progress": 1.0,
            },
        }
        rewards = th.ones(NUM_ENVS)
        for i in reset_envs:
            self.seed[i] = -1
            self.t[i] = 0
        return self._obs(), rewards, dones, extras


def _evaluator(env: _SeededEnv, tmp_path, seeds=range(100, 110), dr_seed=None):
    return PolicyEvaluator(
        env=env, seeds=seeds, out_dir=tmp_path, domain_randomization_seed=dr_seed
    )


def test_wilson_interval():
    low, high = wilson_interval(successes=900, num=1000)
    assert low == pytest.approx(0.8798, abs=1e-3)
    assert high == pytest.approx(0.9170, abs=1e-3)
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert wilson_interval(10, 10)[1] == 1.0


def test_evaluation_runs_every_seed_once_in_the_nominal_domain(tmp_path):
    env = _SeededEnv()
    seen_obs = []

    def policy(obs: th.Tensor) -> th.Tensor:
        seen_obs.append(obs.clone())
        return th.zeros(NUM_ENVS, 2)

    evaluator = _evaluator(env, tmp_path)
    metrics = evaluator.evaluate(policy=policy, iteration=7, titles=["eval"])

    seeds = list(range(100, 110))
    # 3 waves (4 + 4 + 2 episodes), each one starting with the seeded observations
    assert seen_obs[0][:, 0].tolist() == [100, 101, 102, 103]
    assert env.nominal_steps > 0 and not env.in_nominal_domain
    assert env.randomizations == []
    assert metrics["num_episodes"] == 10
    assert metrics["success_rate"] == pytest.approx(0.5)
    assert metrics["time_out_rate"] == 0.0
    assert metrics["mean_episode_length_s"] == pytest.approx(
        np.mean([(s % 5 + 1) * 0.1 for s in seeds])
    )
    assert metrics["mean_final_coverage"] == pytest.approx(np.mean(seeds) / 1000)
    assert "mean_success_rate" not in metrics and "mean_rew_progress" not in metrics

    with (tmp_path / "eval_000007.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["seed"]) for r in rows] == seeds
    assert [r["success"] == "True" for r in rows] == [s % 2 == 0 for s in seeds]
    assert [float(r["final_coverage"]) for r in rows] == pytest.approx(
        [s / 1000 for s in seeds]
    )


def test_seeded_domain_randomization_is_reproducible_and_keeps_the_training_rng(
    tmp_path,
):
    def run(policy_noise: bool) -> _SeededEnv:
        env = _SeededEnv()

        def policy(obs: th.Tensor) -> th.Tensor:
            # another policy consumes the RNG differently, the randomization stays the same
            if policy_noise:
                th.rand(3)
            return th.zeros(NUM_ENVS, 2)

        _evaluator(env, tmp_path, dr_seed=5).evaluate(policy=policy, iteration=0)
        return env

    th.manual_seed(0)
    expected_next = th.rand(1)
    th.manual_seed(0)
    first = run(policy_noise=False)
    assert th.equal(th.rand(1), expected_next)

    second = run(policy_noise=True)
    assert len(first.randomizations) == 3  # one per wave
    assert first.randomizations == second.randomizations
    assert first.observed_noise == second.observed_noise
    assert len(set(first.randomizations)) == 3


def test_domain_randomization_needs_it_configured(tmp_path):
    with pytest.raises(ValueError, match="dr.enabled"):
        _evaluator(_SeededEnv(dr_enabled=False), tmp_path, dr_seed=1)


def test_evaluator_title_suffix(tmp_path):
    assert _evaluator(_SeededEnv(), tmp_path).title_suffix == ""
    assert _evaluator(_SeededEnv(), tmp_path, dr_seed=3).title_suffix == "_dr3"
