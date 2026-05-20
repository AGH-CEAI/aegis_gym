from dataclasses import dataclass, field
from .base_cfg import BaseCfg, ToggleCfg


@dataclass(slots=True, frozen=True)
class CutoutCfg(BaseCfg):
    prob: float = 0.3
    min_size: int = 4
    max_size: int = 12


@dataclass(slots=True, frozen=True)
class ImageAugCfg(ToggleCfg):
    brightness_jitter: float = 0.25
    contrast_jitter: float = 0.20
    gaussian_noise_std: float = 0.025
    gamma_range: float = 0.25
    channel_jitter: float = 0.05
    blur_prob: float = 0.30
    blur_kernel_size: int = 3
    blur_sigma: float = 0.8
    cutout: CutoutCfg = field(default_factory=CutoutCfg)


@dataclass(slots=True, frozen=True)
class PDGainCfg(ToggleCfg):
    kp_noise: float = 0.10
    kv_noise: float = 0.10


@dataclass(slots=True, frozen=True)
class MaxSpeedCfg(ToggleCfg):
    linear_speed_noise: float = 0.03
    angular_speed_noise: float = 0.03


@dataclass(slots=True, frozen=True)
class CameraPoseCfg(BaseCfg):
    translation_std: float = 0.0
    rotation_std_deg: float = 1.0


@dataclass(slots=True, frozen=True)
class CamerasExtrinsicsCfg(ToggleCfg):
    scene_cam: CameraPoseCfg = field(default_factory=CameraPoseCfg)
    tool_cams: CameraPoseCfg = field(default_factory=CameraPoseCfg)


@dataclass(slots=True, frozen=True)
class CameraFovValueCfg(BaseCfg):
    base_fov: float = 38
    std_deg: float = 2.0


@dataclass(slots=True, frozen=True)
class CamerasFovCfg(ToggleCfg):
    scene_cam: CameraFovValueCfg = field(default_factory=CameraFovValueCfg)
    tool_cams: CameraFovValueCfg = field(
        default_factory=lambda: CameraFovValueCfg(base_fov=30)
    )


@dataclass(slots=True, frozen=True)
class DomainRandomizationCfg(ToggleCfg):
    debug_viewer: bool = False

    image_aug: ImageAugCfg = field(default_factory=ImageAugCfg)
    pd_gains: PDGainCfg = field(default_factory=PDGainCfg)
    max_speed: MaxSpeedCfg = field(default_factory=MaxSpeedCfg)

    cameras_extrinsics: CamerasExtrinsicsCfg = field(
        default_factory=CamerasExtrinsicsCfg
    )

    cameras_fov: CamerasFovCfg = field(default_factory=CamerasFovCfg)
