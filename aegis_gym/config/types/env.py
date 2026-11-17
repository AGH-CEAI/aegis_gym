from dataclasses import dataclass

from .base_cfg import BaseCfg
from .enum_types import CamerasSetup

# Genesis `RigidOptions` entries that may be overridden through `rigid_contact`: contact detection and the
# constraint solver. The scene sets the rest itself (dt, solver type, batching, collision and joint limit switches).
RIGID_CONTACT_KEYS: dict[str, type] = {
    "box_box_detection": bool,
    "enable_multi_contact": bool,
    "enable_contact_patch": bool,
    "use_gjk_collision": bool,
    "enable_torsional_friction": bool,
    "enable_rolling_friction": bool,
    "iterations": int,
    "ls_iterations": int,
    "noslip_iterations": int,
    "max_collision_pairs": int,
    "max_contacts": int,
    "constraint_timeconst": float,
    "tolerance": float,
    "ls_tolerance": float,
    "noslip_tolerance": float,
    "impratio": float,
    "contact_pruning_tolerance": float,
}


@dataclass(slots=True)
class EnvCfg(BaseCfg):
    env_name: str
    num_envs: int
    num_obs: int
    num_actions: int
    action_max_linear_speed: float
    action_max_angular_speed: float
    episode_length_s: float
    max_steps: int | None
    ctrl_dt: float
    policy_dt: float
    cameras_setup: CamerasSetup
    image_resolution: tuple[int, int]
    use_rasterizer: bool
    visualize_camera: bool
    visualize_cell: bool
    table_size: list[float]
    workbench_size: list[float]
    env_specific_dict: dict
    # overrides of the Genesis `RigidOptions` (see RIGID_CONTACT_KEYS); None keeps the Genesis defaults
    rigid_contact: dict | None = None

    def __post_init__(self) -> None:
        validate_rigid_contact(self.rigid_contact)


def validate_rigid_contact(overrides: dict | None) -> None:
    """Raises a `ValueError` for keys outside `RIGID_CONTACT_KEYS` or values of the wrong type."""
    errors = []
    for key, value in (overrides or {}).items():
        expected = RIGID_CONTACT_KEYS.get(key)
        if expected is None:
            errors.append(f"unknown key {key!r}")
        elif expected is bool and not isinstance(value, bool):
            errors.append(f"{key!r} must be a bool, got {value!r}")
        elif expected is int and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            errors.append(f"{key!r} must be an int, got {value!r}")
        elif expected is float and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            errors.append(f"{key!r} must be a number, got {value!r}")
    if errors:
        raise ValueError(
            "Invalid `env.rigid_contact`: "
            + "; ".join(errors)
            + f". Allowed keys: {', '.join(RIGID_CONTACT_KEYS)}."
        )
