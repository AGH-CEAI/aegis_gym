import pickle
from dataclasses import dataclass, fields, asdict, is_dataclass
from typing import ClassVar, Any
from pathlib import Path

import torch as th
from clearml import Task
from config_types.domain_randomization import DomainRandomizationCfg


@dataclass(slots=True, frozen=True)
class GraspConfig:
    logger_cfg: dict[str, Any]
    rl_cfg: dict[str, Any]
    bc_cfg: dict[str, Any]
    env_cfg: dict[str, Any]
    robot_cfg: dict[str, Any]
    dr_cfg: DomainRandomizationCfg

    _device: ClassVar["th.device"] = None
    _instance: ClassVar["GraspConfig | None"] = None

    @classmethod
    def set_device(cls, device: "th.device") -> None:
        cls._device = device

    @classmethod
    def get_device(cls) -> "th.device":
        return cls._device

    @classmethod
    def get_instance(cls) -> "GraspConfig":
        if cls._instance is None:
            raise RuntimeError("GraspConfig has not been created.")
        return cls._instance

    @classmethod
    def create(cls) -> "GraspConfig":
        cls._instance = cls(
            logger_cfg=get_logger_cfg(),
            rl_cfg=get_rl_cfg(),
            bc_cfg=get_bc_cfg(),
            env_cfg=get_env_cfg(),
            robot_cfg=get_robot_cfg(),
            dr_cfg=DomainRandomizationCfg.from_dict(get_dr_cfg()),
        )
        return cls._instance

    @classmethod
    def create_with_clearml(cls, task: Task) -> "GraspConfig":
        instance = cls.create()

        values: dict[str, Any] = {}

        for field in fields(instance):
            value = getattr(instance, field.name)

            if is_dataclass(value):
                connected = task.connect_configuration(
                    asdict(value),
                    name=field.name,
                )

                value = type(value).from_dict(connected)

            else:
                value = task.connect_configuration(
                    value,
                    name=field.name,
                )

            values[field.name] = value

        cls._instance = cls(**values)
        return cls._instance

    def to_pickle(self, path: Path) -> None:
        data = {field.name: getattr(self, field.name) for field in fields(self)}
        with path.open("wb") as f:
            pickle.dump(data, f)


def get_logger_cfg() -> dict[str, Any]:
    return {
        # Logger
        "logger": "clearml",  # tensorboard, neptune, wandb, clearml
        "neptune_project": "TEST_PLAYGROUND/aegis_grasp",
        "wandb_project": "TEST_PLAYGROUND/aegis_grasp",
        "clearml_project": "TEST_PLAYGROUND/aegis_grasp",
        "clearml_log_cfg_as_hyperparams": False,
    }


def get_rl_cfg() -> dict[str, Any]:
    return {
        "class_name": "OnPolicyRunner",
        # General
        "num_steps_per_env": 24,  # Number of steps per environment per iteration
        "max_iterations": None,  # Number of policy updates
        "seed": 1,
        # Observations
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },  # Maps observation groups to sets. See `vec_env.py` for more information
        # Logging parameters
        "save_interval": 100,  # Check for potential saves every `save_interval` iterations
        "best_model_skip_iters": 100,  # Ignore first N iterations when tracking best model
        "experiment_name": None,
        "run_name": "",
        "algorithm": {
            "class_name": "PPO",
            # Training
            "learning_rate": 0.0003,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",  # adaptive, fixed
            # Value function
            "value_loss_coef": 1.0,
            "clip_param": 0.2,
            "use_clipped_value_loss": True,
            # Surrogate loss
            "desired_kl": 0.01,
            "entropy_coef": 0.0,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            # Miscellaneous
            "normalize_advantage_per_mini_batch": False,
            # Random network distilation
            "rnd_cfg": None,
            # Symmetry augmentation
            "symmetry_cfg": None,
        },
        "reset_last_layer_weights": {
            "interval": 0,  # Set above 0 to enable
            "part": "both",  # `actor`, `critic` or `both`
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",  # original: "relu"
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "init_noise_std": 1.0,
            "actor_hidden_dims": [128, 128, 64],
            "critic_hidden_dims": [128, 128, 64],
            "noise_std_type": "scalar",  # 'scalar' or 'log'
            "state_dependent_std": False,
            "detach_actor_grad": False,
        },
    }


def get_bc_cfg() -> dict[str, Any]:
    return {
        # basic training parameters
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        "save_recons": False,
        "save_recon_freq": 100,
        "use_teacher_mixing": False,
        # network architecture
        "policy": {
            "encoder_type": "per_camera_cnn",  # shared_cnn, per_camera_cnn, autoencoder
            "fusion_type": "attention_spatial",  # linear, attention_vector, attention_spatial
            "use_pose_head": True,
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
            },
            "vision_encoder_spatial": {
                "conv_layers": [
                    {
                        "in_channels": 3,
                        "out_channels": 8,
                        "kernel_size": 3,
                        "stride": 1,
                        "padding": 1,
                    },
                    {
                        "in_channels": 8,
                        "out_channels": 16,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 16,
                        "out_channels": 32,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                    {
                        "in_channels": 32,
                        "out_channels": 64,
                        "kernel_size": 3,
                        "stride": 2,
                        "padding": 1,
                    },
                ],
            },
            "linear_fusion": {
                "fusion_output_dim": 512,
                "pool_size": 4,
            },
            "attention_vector_fusion": {
                "fusion_output_dim": 512,
                "num_heads": 4,
                "pool_size": 4,
            },
            "attention_spatial_fusion": {
                "fusion_output_dim": 256,
                "num_heads": 4,
            },
            "action_head": {
                "state_obs_dim": 7,  # end-effector pose as additional state observation
                "hidden_dims": [128, 128, 64],
            },
            "pose_head": {
                "hidden_dims": [64, 64],
            },
        },
        # training settings
        "buffer_size": 1000,
        "save_freq": 50,
        "eval_freq": 50,
        "best_model_skip_iters": 100,  # Ignore first N iterations when tracking best model
        "reset_last_layer_weights": {
            "interval": 0,  # Set above 0 to enable
            "part": "all",  # `action`, `pose` or `all`
        },
        "algorithm": {
            "rnd_cfg": None,
        },
    }


def get_task_cfgs() -> tuple[dict[str, Any], dict[str, Any]]:
    env_cfg = get_env_cfg()
    robot_cfg = get_robot_cfg()
    return env_cfg, robot_cfg


def get_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 10,
        "num_obs": 14,
        "num_actions": 6,
        "action_scaling": {
            "max_linear_speed": 0.098,  # m/s
            "max_angular_speed": 0.1,  # rad/s
        },
        # "episode_length_s": 5.0,
        "episode_length_s": 10.0,
        "ctrl_dt": 0.004,  # 1 / 250 Hz (RTDE protocol freq), original 0.01
        "policy_dt": 0.04,  # 1 / 25 Hz, used to calculate number of steps
        "box_sizes": {
            "default": [0.03, 0.08, 0.06],
            "symmetrical": [0.0283, 0.0283, 0.1005],
        },
        "table_size": [0.55, 0.84, 0.818],
        # TODO(issue#98): Move URDF-depended values to the CLearML dataset
        "workbench_size": [0.64, 1.0, 0.821],
        "box_collision": False,
        "box_fixed": True,
        "image_resolution": (64, 64),
        "use_rasterizer": True,
        "visualize_camera": False,
        "visualize_cell": True,
        "camera_setup": "default",  # options: default, scene_dual
        "reward_scales": {
            "keypoints": 1.0,
        },
    }


def get_dr_cfg() -> dict[str, Any]:
    return {
        "enabled": True,
        "debug_viewer": True,
        "image_aug": {
            "enabled": True,
            "per_episode_aug": True,
            "brightness_jitter": 0.4,
            "contrast_jitter": 0.3,
            "gaussian_noise_std": 0.025,
            "gamma_range": 0.25,
            "channel_jitter": 0.05,
            "blur_prob": 0.30,
            "blur_kernel_size": 3,
            "blur_sigma": 0.8,
            "cutout": {
                "prob": 0.3,
                "min_size": 4,
                "max_size": 12,
            },
        },
        "pd_gains": {
            "enabled": True,
            "kp_noise": 0.10,
            "kv_noise": 0.10,
        },
        "max_speed": {
            "enabled": True,
            "linear_speed_noise": 0.03,
            "angular_speed_noise": 0.03,
        },
        "cameras_extrinsics": {
            "enabled": True,
            "scene_cam": {
                "translation_std": 0.0,
                "rotation_std_deg": 1.0,
            },
            "tool_cams": {
                "translation_std": 0.001,
                "rotation_std_deg": 0.6,
            },
        },
    }


def get_robot_cfg() -> dict[str, Any]:
    return {
        "ee_link_name": "robotiq_hande_end",
        "gripper_link_names": [
            "robotiq_hande_left_finger",
            "robotiq_hande_right_finger",
        ],
        "default_arm_dof": [0.0, -2.09, 2.09, -1.57, -1.57, 0.0],
        "default_gripper_dof": [0.025, 0.025],
        "ik_method": "dls_ikv",
        "urdf_model_id": {
            "cell": "c44c56e7671d4004b120b0341fb727a4",
            "cell_collision": "0424f220ddc54091ae3b56b29854532f",
            "no_cell": "718ea536c68c4aaba79d1515ced27eeb",
        },
    }
