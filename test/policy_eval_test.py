import csv
from contextlib import contextmanager

import numpy as np
import pytest
import torch as th

from aegis_gym.config.types import Modality
from aegis_gym.runners.policy_eval import (
    PolicyEvaluator,
    _episode_rows,
    wilson_interval,
)

NUM_ENVS = 4
SEEDS = tuple(range(100, 110))


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


def _evaluator(env: _SeededEnv, tmp_path, seeds=SEEDS, dr_seed=None):
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


def test_episode_rows_take_the_env_flags_from_its_episode_rates():
    dones = th.tensor([True, False, True])
    extras = {
        "success": th.tensor([True, False, False]),
        "time_outs": th.tensor([0.0, 0.0, 1.0]),
        "episode": {
            # per reset env (0 and 2); `success_rate` duplicates `success`
            "success_rate": th.tensor([1.0, 0.0]),
            "tee_off_table_rate": th.tensor([0.0, 1.0]),
            "final_coverage": th.tensor([0.9, 0.2]),
            "rew_progress": 0.5,
        },
    }

    rows = _episode_rows(
        ended=dones,
        dones=dones,
        extras=extras,
        seeds=[7, 8, 9],
        returns=th.tensor([1.0, 2.0, 3.0]),
        lengths=th.tensor([4, 5, 6]),
        policy_dt=0.5,
    )

    assert [row["seed"] for row in rows] == [7, 9]
    assert [row["success"] for row in rows] == [True, False]
    assert [row["time_out"] for row in rows] == [False, True]
    assert [row["tee_off_table"] for row in rows] == [False, True]
    assert [row["final_coverage"] for row in rows] == pytest.approx([0.9, 0.2])
    assert [row["episode_length_s"] for row in rows] == [2.0, 3.0]
    assert "success_rate" not in rows[0] and "rew_progress" not in rows[0]
    assert PolicyEvaluator._summarize(rows)["tee_off_table_rate"] == 0.5


class _TeeEnv(_SeededEnv):
    """The tee at the origin with the yaw 0 in every env."""

    def __init__(self):
        super().__init__()
        self._tee_pose = th.zeros(NUM_ENVS, 7)
        self._tee_pose[:, 3] = 1.0  # wxyz

    def get_modality_observation(self, modality: Modality) -> th.Tensor:
        assert modality == Modality.OBJECT_POSE
        return self._tee_pose


class _DecodingPolicy:
    """Predicts the tee 3 cm off in x (0.3 dm), with the right yaw."""

    def __init__(self):
        self.latent_pose = th.tensor([[0.3, 0.0, 0.0, 1.0]]).repeat(NUM_ENVS, 1)

    def __call__(self, obs: th.Tensor) -> th.Tensor:
        return th.zeros(NUM_ENVS, 2)


def test_latent_pose_errors_are_recorded_per_episode(tmp_path):
    metrics = _evaluator(_TeeEnv(), tmp_path).evaluate(
        policy=_DecodingPolicy(), iteration=0
    )

    assert metrics["mean_latent_pose_err_cm"] == pytest.approx(3.0, abs=1e-4)
    assert metrics["mean_latent_pose_yaw_err_deg"] == pytest.approx(0.0, abs=1e-4)
    with (tmp_path / "policy_eval_000000.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(
        float(r["latent_pose_err_cm"]) == pytest.approx(3.0, abs=1e-4) for r in rows
    )


def test_policies_without_a_latent_pose_record_no_pose_errors(tmp_path):
    metrics = _evaluator(_TeeEnv(), tmp_path).evaluate(
        policy=lambda obs: th.zeros(NUM_ENVS, 2), iteration=0
    )
    assert "mean_latent_pose_err_cm" not in metrics


def test_an_image_condition_is_in_the_titles_and_reads_the_images_through_the_wrappers(
    tmp_path,
):
    nominal = _evaluator(_SeededEnv(), tmp_path)
    blackout = PolicyEvaluator(
        env=_SeededEnv(), seeds=SEEDS, out_dir=tmp_path, condition="blackout_cam_scene"
    )

    assert blackout.title_suffix == "_blackout_cam_scene"
    assert blackout.augments_images and not nominal.augments_images


def test_the_gate_statistics_are_the_mean_over_the_steps(tmp_path):
    weights = iter([0.2, 0.4] * 100)

    class _GatedPolicy:
        def __call__(self, obs: th.Tensor) -> th.Tensor:
            self.weight = next(weights)
            return th.zeros(NUM_ENVS, 2)

        def fusion_stats(self) -> dict[str, float]:
            return {"weight_cam_scene": self.weight}

    metrics = _evaluator(_SeededEnv(), tmp_path).evaluate(
        policy=_GatedPolicy(), iteration=0, titles=["eval"]
    )

    assert metrics["gate_weight_cam_scene"] == pytest.approx(0.3, abs=0.02)
