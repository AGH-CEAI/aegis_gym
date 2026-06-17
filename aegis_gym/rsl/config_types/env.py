from typing import Literal, Optional
from dataclasses import dataclass
from .base_cfg import BaseCfg, ImageShape, Shape3D


@dataclass(slots=True, frozen=True)
class EnvCfg(BaseCfg):
    show_cameras_gui: bool
    use_rasterizer: bool
    show_viewer: bool

    num_envs: int
    num_obs: int
    num_privileged_obs: Optional[int]
    num_actions: int
    rgb_image_shape: ImageShape
    show_cell: bool
    camera_setup: Literal["default", "scene_dual"]
    table_size: Shape3D
    workbench_size: Shape3D
    box_size: Shape3D
    ctrl_dt: float
    policy_dt: float
    sim_substeps: int
    episode_length_s: float
    max_episode_length: int
    max_linear_speed: float
    max_angular_speed: float
    reward_scales: dict
    last_step_ts: Optional[float]
    robot_cfg: dict


@dataclass(slots=True, frozen=True)
class EntityCfg(BaseCfg):
    type: str
    size: Shape3D
    fixed: bool
    collision: bool
    color: Shape3D
