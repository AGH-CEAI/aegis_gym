from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch as th
from rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner

from aegis_gym.config.types import ExpConfig
from aegis_gym.envs import BaseEnv

from .base_runner import BasePolicyRunner


class OnPolicyRunner(BasePolicyRunner):
    def __init__(self, env: BaseEnv, cfg: ExpConfig):
        super().__init__(env=env, cfg=cfg)

        rsl_rl_cfg = cfg.rl_cfg.as_dict()
        rsl_rl_cfg.update(cfg.logger_cfg.as_dict())
        log_dir = cfg.logger_cfg.local_log_dir
        self.runner = RslRlOnPolicyRunner(
            env=env,
            train_cfg=rsl_rl_cfg,
            log_dir=str(log_dir),
            device=str(cfg.get_device()),
        )
        if cfg.logger_cfg.logger == "clearml":
            self._drop_time_scalars()

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        self.runner.learn(
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def _drop_time_scalars(self) -> None:
        """
        rsl_rl logs the `*/time` scalars with the elapsed seconds as the step, which ClearML takes
        as the iteration: e.g. 3000 iterations over 5 h would log up to "iteration" 18000, also
        moving the task's last iteration (and its resource monitor) there. ClearML plots the scalars
        against the wall time itself, so these duplicates are dropped.
        """
        writer = self.runner.logger.writer
        if writer is None:
            return
        add_scalar = writer.add_scalar

        def add_scalar_per_iteration(tag: str, *args: Any, **kwargs: Any) -> None:
            if tag.endswith("/time"):
                return
            add_scalar(tag, *args, **kwargs)

        writer.add_scalar = add_scalar_per_iteration

    def learn_in_chunks(
        self,
        num_learning_iterations: int,
        chunk_iterations: int,
        on_chunk_end: Callable[[int], None],
        init_at_random_ep_len: bool = False,
    ) -> None:
        """
        Runs `learn()` in chunks of `chunk_iterations`, calling `on_chunk_end(iteration)` between
        them (not after the last one). `learn()` fetches fresh observations on every call, so the
        callback may step and reset the env.
        """
        remaining = num_learning_iterations
        while remaining > 0:
            chunk = min(chunk_iterations, remaining)
            self.learn(
                num_learning_iterations=chunk,
                init_at_random_ep_len=init_at_random_ep_len,
            )
            remaining -= chunk
            if remaining > 0:
                on_chunk_end(self.current_iteration)
                # `learn()` leaves the counter at its last iteration, continue with the next one
                self.runner.current_learning_iteration += 1

    @property
    def current_iteration(self) -> int:
        return self.runner.current_learning_iteration

    def save(self, path: Path, infos: dict | None = None) -> None:
        self.runner.save(path=str(path), infos=infos)

    def load(self, path: Path) -> None:
        self.runner.load(path=str(path))

    def get_inference_policy(self, device: th.device | None = None) -> Any:
        device = device or th.device("cpu")
        return self.runner.get_inference_policy(device=str(device))

    def export_policy(self, path: Path, filename: str = "policy.pt") -> None:
        try:
            if filename.endswith(".pt"):
                return self.runner.export_policy_to_jit(path=path, filename=filename)
            if filename.endswith(".onnx"):
                return self.runner.export_policy_to_onnx(path=path, filename=filename)
        except (ValueError, OSError) as e:
            raise NotImplementedError("Export policy not implemented.") from e
