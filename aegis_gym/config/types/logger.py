from dataclasses import dataclass, field
from pathlib import Path

from .base_cfg import BaseCfg

POLICY_PREVIEW_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


@dataclass(slots=True)
class LoggerCfg(BaseCfg):
    logger: str
    neptune_project: str
    wandb_project: str
    clearml_project: str
    clearml_log_cfg_as_hyperparams: bool
    local_log_dir: Path
    policy_preview_interval: int = 0
    policy_preview_resolution: list[int] = field(default_factory=lambda: [320, 240])
