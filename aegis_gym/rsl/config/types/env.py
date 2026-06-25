from dataclasses import dataclass
from typing import Literal
from .base_cfg import BaseCfg


@dataclass(slots=True)
class EnvCfg(BaseCfg):
    num_envs: int
    num_obs: int
    num_actions: int
    action_max_linear_speed: float
    action_max_angular_speed: float
    episode_length_s: float
    ctrl_dt: float
    policy_dt: float
    # TODO introduce size config
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
    # TODO consider changing camera setup to cameras_num
    camera_setup: Literal["default", "scene_dual"]
    reward_scales: dict
