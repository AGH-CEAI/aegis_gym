import csv
import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch as th
from clearml import Task
from tqdm import tqdm

from aegis_gym.aux.logging import get_logger
from aegis_gym.envs import BaseEnv
from aegis_gym.envs.wrappers import BaseEnvWrapper, innermost_env

from .policy_preview import InferencePolicy

EpisodeRow = dict[str, float | int | bool]


def wilson_interval(successes: int, num: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of a success rate (95 % with the default `z`)."""
    if num == 0:
        return 0.0, 1.0
    p = successes / num
    denom = 1.0 + z * z / num
    center = (p + z * z / (2 * num)) / denom
    half = z * math.sqrt(p * (1.0 - p) / num + z * z / (4 * num * num)) / denom
    return max(0.0, center - half), min(1.0, center + half)


class PolicyEvaluator:
    """
    Success rate of a policy over a fixed list of seeded initial states (`reset_seeded()`), with
    the per episode results saved as a CSV (and a ClearML artifact) for the paired comparisons.

    The episodes run in waves of `num_envs`, started together. By default in the nominal domain
    (without any domain randomization). With the `domain_randomization_seed`, each wave gets the
    configured domain randomization drawn from that seed: the scene's randomizations and, with a
    `VisionAugEnvWrapper` in the env, the image augmentation (read the images through `env` then,
    see `make_visual_bc_policy(augment_images=True)`). The randomization is the same for every
    evaluated policy at the same `num_envs` (the domain randomization applies to all the envs of
    a wave at once); the training's torch and numpy RNG states are restored afterwards.

    All the envs are reset afterwards, so it can run between the training iterations.
    """

    def __init__(
        self,
        env: BaseEnv,
        seeds: Sequence[int],
        out_dir: Path,
        domain_randomization_seed: int | None = None,
        report_task: Task | None = None,
    ):
        self._env = env
        self._base = innermost_env(env)
        self._seeds = list(seeds)
        self._out_dir = Path(out_dir)
        self._dr_seed = domain_randomization_seed
        self._report_task = report_task
        self._logger = get_logger("PolicyEval")
        if not self._seeds:
            raise ValueError("The policy evaluation needs at least one seed.")
        if self.randomizes_domain and not self._base.is_domain_randomization_enabled():
            raise ValueError(
                "The evaluation with the domain randomization needs it enabled (`dr.enabled`)."
            )

    @staticmethod
    def is_supported(env: BaseEnv) -> bool:
        """Whether the env has the seeded reset the evaluation relies on."""
        env_cls = type(innermost_env(env))
        return env_cls._reset_idx_seeded is not BaseEnv._reset_idx_seeded

    @property
    def randomizes_domain(self) -> bool:
        return self._dr_seed is not None

    @property
    def title_suffix(self) -> str:
        """`_dr<seed>` with the domain randomization, so its results aren't mixed with nominal."""
        return f"_dr{self._dr_seed}" if self.randomizes_domain else ""

    @property
    def num_episodes(self) -> int:
        return len(self._seeds)

    def evaluate(
        self,
        policy: InferencePolicy,
        iteration: int,
        titles: Sequence[str] = ("policy_eval",),
        single_values: bool = False,
    ) -> dict[str, float]:
        """
        Runs the episodes, reports the metrics under each of the `titles` (also as the ClearML
        single values with `single_values`, e.g. for the final evaluation) and returns them.
        """
        domain = (
            f"domain randomization seed {self._dr_seed}"
            if self.randomizes_domain
            else "nominal domain"
        )
        self._logger.info(
            f"Evaluating {len(self._seeds)} episodes ({domain}), "
            f"seeds {self._seeds[0]}..{self._seeds[-1]}"
        )
        start = time.time()
        with th.inference_mode():
            with self._base.nominal_domain(), self._forked_rng():
                rows = self._run_episodes(policy=policy, desc=titles[0])
            self._env.reset()
        metrics = self._summarize(rows)
        metrics["duration_s"] = time.time() - start
        for title in titles:
            self._report(
                title=title,
                iteration=iteration,
                metrics=metrics,
                rows=rows,
                single_values=single_values,
            )
        return metrics

    @contextmanager
    def _forked_rng(self) -> Iterator[None]:
        if not self.randomizes_domain:
            yield
            return
        device = self._base.device
        devices = (
            [device.index if device.index is not None else th.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        np_state = np.random.get_state()
        with th.random.fork_rng(devices=devices):
            try:
                yield
            finally:
                np.random.set_state(np_state)

    def _reseed(self, *keys: int) -> None:
        """Reseeds the torch and numpy RNGs from the domain randomization seed and the `keys`."""
        if not self.randomizes_domain:
            return
        seed = self._dr_seed
        for key in keys:
            seed = (seed * 1_000_003 + key + 1) % (1 << 62)
        th.manual_seed(seed)
        np.random.seed(seed % (1 << 32))

    def _randomize_wave_domain(self, wave: int) -> None:
        if not self.randomizes_domain:
            return
        self._reseed(wave)
        base = self._base
        base.apply_domain_randomization(th.arange(base.num_envs, device=base.device))
        env = self._env
        while True:
            resample = getattr(type(env), "resample_augmentation", None)
            if resample is not None:
                env.resample_augmentation()
            if not isinstance(env, BaseEnvWrapper):
                break
            env = env.unwrapped

    def _run_episodes(self, policy: InferencePolicy, desc: str) -> list[EpisodeRow]:
        base = self._base
        num_envs = base.num_envs
        max_steps = int(base.max_episode_length) + 1
        policy_dt = base.get_policy_dt()
        rows: list[EpisodeRow] = []

        base.reset()
        progress = tqdm(total=len(self._seeds), desc=desc, unit="episode")
        with progress:
            for wave, first in enumerate(range(0, len(self._seeds), num_envs)):
                seeds = self._seeds[first : first + num_envs]
                num = len(seeds)
                # randomized before the seeded reset, so its observations show the new domain
                self._randomize_wave_domain(wave)
                obs = base.reset_seeded(
                    envs_idx=th.arange(num, device=base.device), seeds=seeds
                )
                active = th.ones(num, dtype=th.bool, device=base.device)
                returns = th.zeros(num, device=base.device)
                lengths = th.zeros(num, dtype=th.long, device=base.device)

                for step in range(max_steps):
                    self._reseed(wave, 2 * step + 1)
                    actions = policy(obs)
                    self._reseed(wave, 2 * step + 2)
                    obs, rewards, dones, extras = base.step(actions)

                    returns += rewards[:num] * active
                    lengths += active.long()
                    ended = dones[:num].bool() & active
                    if ended.any():
                        rows += _episode_rows(
                            ended=ended,
                            dones=dones.bool(),
                            extras=extras,
                            seeds=seeds,
                            returns=returns,
                            lengths=lengths,
                            policy_dt=policy_dt,
                        )
                        progress.update(int(ended.sum()))
                    active &= ~ended
                    if not active.any():
                        break
                # not ended within the episode length (shouldn't happen): counted as time outs
                for i in th.nonzero(active).flatten().tolist():
                    rows.append(
                        {
                            "seed": seeds[i],
                            "success": False,
                            "time_out": True,
                            "return": float(returns[i]),
                            "episode_length_s": int(lengths[i]) * policy_dt,
                        }
                    )
                    progress.update()
        return sorted(rows, key=lambda row: row["seed"])

    @staticmethod
    def _summarize(rows: list[EpisodeRow]) -> dict[str, float]:
        num = len(rows)
        successes = sum(bool(row["success"]) for row in rows)
        low, high = wilson_interval(successes, num)
        metrics = {
            "num_episodes": float(num),
            "success_rate": successes / num,
            "success_rate_ci95_low": low,
            "success_rate_ci95_high": high,
        }
        keys = dict.fromkeys(k for row in rows for k in row)
        for key in keys:
            if key in ("seed", "success"):
                continue
            values = [row[key] for row in rows if key in row]
            is_flag = isinstance(values[0], bool)
            values = [float(v) for v in values]
            name = f"{key}_rate" if is_flag else f"mean_{key}"
            metrics[name] = float(np.mean(values))
        success_lengths = [
            float(row["episode_length_s"]) for row in rows if row["success"]
        ]
        if success_lengths:
            metrics["mean_success_time_s"] = float(np.mean(success_lengths))
        return metrics

    def _report(
        self,
        title: str,
        iteration: int,
        metrics: dict[str, float],
        rows: list[EpisodeRow],
        single_values: bool,
    ) -> None:
        self._logger.info(
            f"{title} (iteration {iteration}): "
            + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        )
        self._out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._out_dir / f"{title}_{iteration:06d}.csv"
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        task = self._report_task or Task.current_task()
        if task is None:
            self._logger.warning(
                f"No ClearML task, the evaluation is only saved in {csv_path}."
            )
            return
        clearml_logger = task.get_logger()
        for name, value in metrics.items():
            clearml_logger.report_scalar(
                title=title, series=name, value=value, iteration=iteration
            )
            if single_values:
                clearml_logger.report_single_value(name=f"{title}/{name}", value=value)
        self._upload_csv(task=task, csv_path=csv_path)

    def _upload_csv(self, task: Task, csv_path: Path) -> None:
        """
        A finished task takes no new artifacts (its scalars are still accepted): the CSV goes then
        to the running task (e.g. the evaluation's own), named after the reported task's id.
        """
        current = Task.current_task()
        status = task.get_status()
        if (
            status in ("created", "in_progress")
            or current is None
            or current.id == task.id
        ):
            task.upload_artifact(name=csv_path.stem, artifact_object=str(csv_path))
            return
        name = f"{task.id}_{csv_path.stem}"
        current.upload_artifact(name=name, artifact_object=str(csv_path))
        self._logger.info(
            f"The task {task.id} is {status}: the CSV is the artifact `{name}` of {current.id}."
        )


def _episode_rows(
    ended: th.Tensor,
    dones: th.Tensor,
    extras: dict,
    seeds: Sequence[int],
    returns: th.Tensor,
    lengths: th.Tensor,
    policy_dt: float,
) -> list[EpisodeRow]:
    """
    The results of the `ended` episodes of the wave. The env's `extras["episode"]` metrics are of
    the envs reset in this step (all the `dones`, in the order of their indices); the ones per
    episode are kept, without the `*_rate` ones (the same as the flags).
    """
    reset_envs = th.nonzero(dones).flatten().tolist()
    episode = extras.get("episode", {})
    per_episode = {
        key: value
        for key, value in episode.items()
        if isinstance(value, th.Tensor)
        and value.numel() == len(reset_envs)
        and not key.endswith("_rate")
    }

    def flag(key: str, i: int) -> bool | None:
        value = extras.get(key)
        return bool(value[i]) if isinstance(value, th.Tensor) else None

    rows = []
    for i in th.nonzero(ended).flatten().tolist():
        row: EpisodeRow = {
            "seed": seeds[i],
            "success": bool(flag("success", i)),
            "time_out": bool(flag("time_outs", i)),
        }
        tee_off_table = flag("tee_off_table", i)
        if tee_off_table is not None:
            row["tee_off_table"] = tee_off_table
        row["return"] = float(returns[i])
        row["episode_length_s"] = int(lengths[i]) * policy_dt
        position = reset_envs.index(i)
        for key, value in per_episode.items():
            row[key] = float(value.flatten()[position])
        rows.append(row)
    return rows
