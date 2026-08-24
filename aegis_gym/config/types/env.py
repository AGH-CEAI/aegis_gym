from dataclasses import dataclass

from .base_cfg import BaseCfg
from .enum_types import CamerasSetup, EnvType


@dataclass(slots=True)
class EnvCfg(BaseCfg):
    env_type: EnvType
    num_envs: int
    num_obs: int
    num_actions: int
    action_max_linear_speed: float
    action_max_angular_speed: float
    episode_length_s: float
    max_steps: int | None
    ctrl_dt: float
    policy_dt: float
    # TODO(issue#111) introduce size config
    box_size_default: list[float]
    box_size_symmetrical: list[float]
    table_size: list[float]
    workbench_size: list[float]
    box_collision: bool
    box_fixed: bool
    image_resolution: tuple[int, int]
    use_rasterizer: bool
    visualize_camera: bool
    visualize_cell: bool
    # TODO(issue#111) consider changing camera setup to cameras_num
    cameras_setup: CamerasSetup
    reacher_reward_scales: dict
    # Push-T task
    tee_spawnbox_xlength: float
    tee_spawnbox_ylength: float
    tee_spawnbox_xoffset: float
    tee_spawnbox_yoffset: float
    tee_goal_offset: list[float]
    tee_goal_z_rot_deg: float
    tee_success_intersection_thresh: float
    tee_mask_resolution: int
    tee_mask_half_width: float
    tee_friction: float
    push_t_reward_scales: dict
