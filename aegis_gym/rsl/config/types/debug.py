from dataclasses import dataclass
from pathlib import Path

from .base_cfg import BaseCfg


@dataclass(slots=True)
class DebugCfg(BaseCfg):
    enabled: bool = False
    enable_vis_preview: bool = False
    vis_preview_side: int = 256
    vis_preview_max_envs: int = 5
    enable_record_obs: bool = False
    record_dir: Path = Path("/tmp")
