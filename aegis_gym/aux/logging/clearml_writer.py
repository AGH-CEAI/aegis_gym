import os
import pathlib
from dataclasses import asdict
from typing import Any

from clearml import Task
from rsl_rl.utils import LogWriter
from torch.utils.tensorboard import SummaryWriter


class ClearMLLogWriter(LogWriter):
    def __init__(
        self,
        log_dir: str,
        project_name: str,
        log_cfg_as_hyperparams: bool = True,
        flush_secs: int = 10,
    ) -> None:
        self._tb = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
        self._log_cfg_as_hyperparams = log_cfg_as_hyperparams

        self.task = Task.current_task() or Task.init(
            project_name=project_name,
            task_name=os.path.split(log_dir)[-1],
            auto_connect_frameworks={"tensorboard": False},
            reuse_last_task_id=False,
        )

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        if tag.endswith("/time"):
            return

        self._tb.add_scalar(tag, scalar_value, global_step=global_step)
        self.task.get_logger().report_scalar(
            tag, "series", scalar_value, iteration=global_step
        )

    def store_config(self, env_cfg: dict | Any, train_cfg: dict) -> None:
        if not self._log_cfg_as_hyperparams:
            return

        runner_cfg = dict(train_cfg)
        for section in ("actor", "critic", "algorithm"):
            runner_cfg.pop(section, None)

        if isinstance(env_cfg, dict):
            env_dict = env_cfg
        else:
            env_dict = (
                env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
            )

        self.task.connect(runner_cfg, name="runner_cfg")
        self.task.connect(train_cfg.get("actor", {}), name="actor_cfg")
        self.task.connect(train_cfg.get("critic", {}), name="critic_cfg")
        self.task.connect(train_cfg.get("algorithm", {}), name="alg_cfg")
        self.task.connect(env_dict, name="env_cfg")

    def save_model(self, model_path: str, it: int) -> None:
        self.task.upload_artifact(
            name=pathlib.Path(model_path).stem, artifact_object=model_path
        )

    def save_file(self, path: str) -> None:
        self.task.upload_artifact(name=os.path.basename(path), artifact_object=path)

    def save_video(self, video: pathlib.Path, it: int) -> None:
        self.task.get_logger().report_media(
            title="policy", series="rollout", iteration=it, local_path=str(video)
        )

    def stop(self) -> None:
        self._tb.flush()
