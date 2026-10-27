from typing import Any

from aegis_gym.config.types import LoggerCfg

CLEARML_WRITER = "aegis_gym.aux.logging.clearml_writer:ClearMLLogWriter"


def resolve_log_writer_cfg(logger_cfg: LoggerCfg) -> str | dict[str, Any]:
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
