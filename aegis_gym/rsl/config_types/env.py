from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from .base_cfg import BaseCfg


@dataclass(slots=True)
class EnvCfg(BaseCfg):
    num_envs: int
    num_obs: int
    num_actions: 6
    max_linear_speed: float
    max_angulat_speed: float
    episode_length_s: float
    ctrl_dt: float
    policy_dt: float
    box_sizes: None
        # TODO
        default: list
        symmetrical: list
    table_size: list
    workbench_size: list
    box_collision: bool
    box_fixed: bool
    image_resolution: tuple
    use_rasterizer: bool
    visualize_camera: bool
    visualize_cell: bool
    camera_setup: Literal["default", "scene_dual"]
    reward_scales: dict
