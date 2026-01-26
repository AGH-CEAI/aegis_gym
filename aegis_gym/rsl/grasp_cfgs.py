def get_rl_cfg(exp_name: str, max_iterations: int) -> dict:
    # stage 1: privileged reinforcement learning
    return {
        "class_name": "OnPolicyRunner",
        # General
        "num_steps_per_env": 24,  # Number of steps per environment per iteration
        "max_iterations": max_iterations,  # Number of policy updates
        "seed": 1,
        # Observations
        "obs_groups": {
            "policy": ["policy"],
            "critic": ["policy"],
        },  # Maps observation groups to sets. See `vec_env.py` for more information
        # Logging parameters
        "save_interval": 100,  # Check for potential saves every `save_interval` iterations
        "experiment_name": exp_name,
        "run_name": "",
        # Logger
        "logger": "clearml",  # tensorboard, neptune, wandb, clearml
        "neptune_project": "TEST_PLAYGROUND/aegis_grasp",
        "wandb_project": "TEST_PLAYGROUND/aegis_grasp",
        "clearml_project": "TEST_PLAYGROUND/aegis_grasp",
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
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",  # original: "relu"
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "noise_std_type": "scalar",  # 'scalar' or 'log'
            "state_dependent_std": False,
        },
    }


def get_bc_cfg() -> dict:
    # stage 2: vision-based behavior cloning
    return {
        # basic training parameters
        "num_steps_per_env": 24,
        "learning_rate": 0.001,
        "num_epochs": 5,
        "num_mini_batches": 10,
        "max_grad_norm": 1.0,
        # network architecture
        "policy": {
            "vision_encoder": {
                "conv_layers": [
                    {
                        "in_channels": 3,  # 3 channel for rgb image
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
                "pooling": "adaptive_avg",
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
        "log_freq": 10,
        "save_freq": 50,
        "eval_freq": 50,
    }


def get_task_cfgs():
    env_cfg = {
        "num_envs": 10,
        "num_obs": 14,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "sim_substeps": 2,  # 2 or 32
        "box_size": [0.03, 0.08, 0.06],
        "table_size": [0.55, 0.84, 0.82],
        "workbench_size": [0.64, 1.0, 0.806],
        "box_collision": False,
        "box_fixed": True,
        "image_resolution": (64, 64),
        "use_rasterizer": False,
        "visualize_camera": False,
        "visualize_cell": False,
        "camera_setup": "default",  # options: default, scene_dual
    }
    reward_scales = {
        "keypoints": 1.0,
    }
    # robot specific
    robot_cfg = {
        "ee_link_name": "robotiq_hande_end",
        "gripper_link_names": [
            "robotiq_hande_left_finger",
            "robotiq_hande_right_finger",
        ],
        "default_arm_dof": [0.0, -2.09, 2.09, -1.57, -1.57, 0.0],
        "default_gripper_dof": [0.025, 0.025],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_scales, robot_cfg
