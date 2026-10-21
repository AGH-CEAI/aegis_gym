from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch as th
from rsl_rl.runners import OnPolicyRunner as RslRlOnPolicyRunner

from aegis_gym.config.types import ExpConfig, LoggerCfg
from aegis_gym.envs import BaseEnv

from .base_runner import BasePolicyRunner
from .training_logger import TrainingLogger

CLEARML_WRITER = "aegis_gym.aux.logging.clearml_writer:ClearMLLogWriter"


class OnPolicyRunner(BasePolicyRunner):
    def __init__(self, env: BaseEnv, cfg: ExpConfig):
        super().__init__(env=env, cfg=cfg)

        rsl_rl_cfg = cfg.rl_cfg.as_dict()
        rsl_rl_cfg.update(cfg.logger_cfg.as_dict())
        rsl_rl_cfg["logger"] = _resolve_log_writer_cfg(cfg.logger_cfg)
        log_dir = cfg.logger_cfg.local_log_dir
        self.runner = RslRlOnPolicyRunner(
            env=env,
            train_cfg=rsl_rl_cfg,
            log_dir=str(log_dir),
            device=str(cfg.get_device()),
        )

        reset_cfg = cfg.rl_cfg.reset_last_layer_weights
        self._reset_last_layer_part = reset_cfg.get("part", "both")
        self.runner.logger = TrainingLogger(
            log_dir=str(log_dir),
            cfg=rsl_rl_cfg,
            env_cfg=env.get_cfg_as_dict(),
            num_envs=env.num_envs,
            is_distributed=self.runner.is_distributed,
            gpu_world_size=self.runner.gpu_world_size,
            gpu_global_rank=self.runner.gpu_global_rank,
            device=str(cfg.get_device()),
            save_best_model=self.save,
            best_model_skip_iters=cfg.rl_cfg.best_model_skip_iters,
            reset_last_layer=self._reset_last_layer_weights,
            reset_last_layer_interval=reset_cfg.get("interval", 0),
        )

    def learn(
        self, num_learning_iterations: int, init_at_random_ep_len: bool = False
    ) -> None:
        self.runner.learn(
            num_learning_iterations=num_learning_iterations,
            init_at_random_ep_len=init_at_random_ep_len,
        )

    def learn_in_chunks(
        self,
        num_learning_iterations: int,
        chunk_iterations: int,
        on_chunk_end: Callable[[int], None],
        init_at_random_ep_len: bool = False,
    ) -> None:
        remaining = num_learning_iterations
        while remaining > 0:
            chunk = min(chunk_iterations, remaining)
            self.learn(
                num_learning_iterations=chunk,
                init_at_random_ep_len=init_at_random_ep_len,
            )
            remaining -= chunk
            if remaining > 0:
                on_chunk_end(self.current_iter)
                self._discard_interrupted_episodes()
                # `learn()` leaves the counter at its last iteration, continue with the next one
                self.runner.current_learning_iteration += 1

    def _discard_interrupted_episodes(self) -> None:
        logger = self.runner.logger
        for name in (
            "cur_reward_sum",
            "cur_episode_length",
            "cur_ereward_sum",  # with RND only
            "cur_ireward_sum",  # with RND only
        ):
            buffer = getattr(logger, name, None)
            if buffer is not None:
                buffer.zero_()
        alg = self.runner.alg
        alg.actor.reset()
        alg.critic.reset()

    def _reset_last_layer_weights(self) -> None:
        part = self._reset_last_layer_part
        if part not in {"actor", "critic", "both"}:
            raise ValueError(
                f"Last layer reset target must be actor, critic or both, got {part!r}"
            )

        alg = self.runner.alg
        models = {"actor": alg.get_policy(), "critic": alg._raw_critic}
        for name, model in models.items():
            if part in {name, "both"}:
                if not hasattr(model, "reset_last_layer_weights"):
                    raise AttributeError(
                        f"The {name} model cannot reset its last layer. Select a model "
                        f"class that supports it in the configuration."
                    )
                model.reset_last_layer_weights()

    def save(self, path: Path, infos: dict | None = None) -> None:
        self.runner.save(path=str(path), infos=infos)

    def load(self, path: Path) -> None:
        self.runner.load(path=str(path), map_location=str(self.runner.device))
        alg = self.runner.alg
        alg.learning_rate = alg.optimizer.param_groups[0]["lr"]

    @property
    def current_iter(self) -> int:
        return self.runner.current_learning_iteration

    def get_inference_policy(self, device: th.device | None = None) -> Any:
        device = device or th.device("cpu")
        return self.runner.get_inference_policy(device=str(device))

    def export_policy(self, path: Path, filename: str = "policy.pt") -> None:
        if filename.endswith(".pt"):
            return self.runner.export_policy_to_jit(path=str(path), filename=filename)
        if filename.endswith(".onnx"):
            return self.runner.export_policy_to_onnx(path=str(path), filename=filename)
        raise NotImplementedError(f"Unsupported export format: {filename}")


def _resolve_log_writer_cfg(logger_cfg: LoggerCfg) -> str | dict[str, Any]:
    match logger_cfg.logger:
        case "clearml":
            return {
                "class_name": CLEARML_WRITER,
                "project_name": logger_cfg.clearml_project,
                "log_cfg_as_hyperparams": logger_cfg.clearml_log_cfg_as_hyperparams,
            }
        case "wandb":
            return {
                "class_name": "WandbLogWriter",
                "project_name": logger_cfg.wandb_project,
            }
        case "neptune":
            return {
                "class_name": "NeptuneLogWriter",
                "project_name": logger_cfg.neptune_project,
            }
        case other:
            return other
