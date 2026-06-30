from dataclasses import dataclass
from pathlib import Path

from .base_cfg import BaseCfg


@dataclass(slots=True)
class DebugCfg(BaseCfg):
    enabled: bool = False
    swap_tool_cameras: bool = False
    enable_vis_preview: bool = False
    enable_record_obs: bool = False
    record_dir: Path = Path("/tmp")
