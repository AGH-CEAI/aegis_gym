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

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        self.runner.learn(
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )

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
