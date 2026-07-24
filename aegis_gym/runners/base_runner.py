from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch as th

from aegis_gym.envs import BaseEnv
from aegis_gym.config.types import ExpConfig, RLCfg, LoggerCfg


class BasePolicyRunner(ABC):
    """
    An abstract interface to ensure compliance with the rsl_rl's `OnPolicyRunner`.

    Mismatch with the `OnPolicyRunner`:
        * `export_policy_to_jit()`
        * `export_policy_to_onnx()`
        * `_configure_multi_gpu()`
    """

    def __init__(self, env: BaseEnv, cfg: ExpConfig):
        """
        Construct the runner, given algorithm and experiment logger.
        """
        self.env = env
        self.cfg_train: RLCfg = cfg.rl_cfg
        self.cfg_logger: LoggerCfg = cfg.logger_cfg
        self.device = cfg.get_device()

    @abstractmethod
    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        """
        Run the learning loop for specified `num_learning_iterations`.
        """
        ...

    @abstractmethod
    def save(self, path: Path, infos: dict | None = None) -> None:
        """
        Save the models and training state to a given `path` (and upload them via logger).
        """
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load the models and training state from `path`.
        """
        ...

    @abstractmethod
    def get_inference_policy(self, device: th.device) -> Any:
        """
        Return the policy for the inference on the requested device.
        """
        ...

    @abstractmethod
    def export_policy(self, path: Path, filename: str = "policy.pt") -> None:
        """
        Export the model into a given file (and format).
        The rsl_rl's `OnPolicyRunner` implements `export_policy_to_jit()` and `export_policy_to_onnx()`.
        """
        ...
