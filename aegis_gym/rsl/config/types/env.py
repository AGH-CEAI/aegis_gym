from dataclasses import dataclass
from typing import Optional

from .base_cfg import BaseCfg
from .enum_types import CamerasSetup


@dataclass(slots=True)
class EnvCfg(BaseCfg):
    num_envs: int
    num_obs: int
    num_actions: int
    action_max_linear_speed: float
    action_max_angular_speed: float
    episode_length_s: float
    max_steps: Optional[int]
    ctrl_dt: float
    policy_dt: float
    # TODO(issue#111) introduce size config
    box_size_default: list
    box_size_symmetrical: list
    table_size: list
    workbench_size: list
    box_collision: bool
    box_fixed: bool
    image_resolution: tuple[int, int]
    use_rasterizer: bool
    visualize_camera: bool
    visualize_cell: bool
    # TODO(issue#111) consider changing camera setup to cameras_num
    cameras_setup: CamerasSetup
    reward_scales: dict
