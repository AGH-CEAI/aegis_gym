from dataclasses import dataclass
from pathlib import Path

from .base_cfg import BaseCfg


@dataclass(slots=True)
class LoggerCfg(BaseCfg):
    logger: str
    neptune_project: str
    wandb_project: str
    clearml_project: str
    clearml_log_cfg_as_hyperparams: bool

    @classmethod
    def from_yaml(cls, path: Path) -> "LoggerCfg":
        d = cls._load_yaml(path, "logger")
        return cls.from_dict(data=d)
