from dataclasses import dataclass

from .base_cfg import BaseCfg


@dataclass(slots=True)
class LoggerCfg(BaseCfg):
    logger: str
    neptune_project: str
    wandb_project: str
    clearml_project: str
    clearml_log_cfg_as_hyperparams: bool
