# Original implementation by Jakub Płachno (sivral) 2025
import math
from typing import Literal, Optional

import genesis as gs
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from ..robot import RobotCommanderType


ENV_CFG = {
    "num_actions": 6,
    "default_joint_angles": {
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -2.10,
        "elbow_joint": 2.10,
        "wrist_1_joint": -1.57,
        "wrist_2_joint": -1.57,
        "wrist_3_joint": 0.0,
        # "robotiq_hande_left_finger_joint": 0.025,
        # "robotiq_hande_right_finger_joint": 0.025,
    },
    "dof_names": [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        # 'robotiq_hande_left_finger_joint',
        # 'robotiq_hande_right_finger_joint',
    ],
    "kp": [600, 600, 400, 400, 200, 200],
    "kd": [60, 60, 40, 40, 20, 20],
    "robot_pos": [0.0, 0.0, 0.0],
    "table_pos": [0.0, 0.6, 0.41],
    "object_spawn_x": [-0.36, -0.24],
    "object_spawn_y": [0.34, 0.66],
    "object_spawn_z": [0.84, 0.85],
    "target_pos": [-0.1, 0.76, 0.84],
    "target_threshold": 0.04,
    "episode_length_s": 5.0,
    "dt": 0.05,
    "action_scale": 0.5,
    "clip_actions": 100.0,
    "num_obs": 21,
    "obs_scales": {
        "dof_pos": 1.0,
        "dof_vel": 0.1,
    },
    "reward_scales": {
        "near": -0.5,
        "dist": -1.0,
        "control": -0.1,
    },
}


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * th.rand(size=shape, device=device) + lower


class AegisGenesisPusherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[Literal["human", "rgb_array"]] = None,
        reward_type: str = "dense",
        control_type: str = "joints",
        device: str = "cuda",
        robot_interface: RobotCommanderType = RobotCommanderType.REAL,
        cfg: dict = ENV_CFG,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.device = th.device(device)

        show_viewer = False
        if render_mode == "human":
            show_viewer = True

        self.max_episode_length = math.ceil(cfg["episode_length_s"] / cfg["dt"])

        self.num_obs = cfg["num_obs"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.num_actions = cfg["num_actions"]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.target_threshold = cfg["target_threshold"]

        self.obs_scales = cfg["obs_scales"]
        self.reward_scales = cfg["reward_scales"]

        # TODO apply this to Simulation/Real
        self.target = self.scene.add_entity(
            gs.morphs.Cylinder(
                height=0.00001, radius=0.04, pos=(-0.1, 0.76, 0.82), fixed=True
            ),
            surface=gs.surfaces.Default(color=(1.0, 0.0, 0.0)),
            material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
        )

        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=(0.04, 0.04, 0.04),
                pos=(0.0, 0.7, 0.84),
            ),
            surface=gs.surfaces.Default(color=(1.0, 1.0, 1.0)),
            material=gs.materials.Rigid(rho=8000.0, friction=0.6, coup_friction=0.6),
        )

        self.scene.build()

        self.episode_step = 0.0

        self.motor_dofs = [
            self.robot.get_joint(name).dof_idx_local for name in ENV_CFG["dof_names"]
        ]
        self.robot.set_dofs_kp(ENV_CFG["kp"], self.motor_dofs)
        self.robot.set_dofs_kv(ENV_CFG["kd"], self.motor_dofs)

        self.default_dof_pos = th.tensor(
            [ENV_CFG["default_joint_angles"][name] for name in ENV_CFG["dof_names"]],
            device=self.device,
        )
        self.dof_pos = th.zeros(self.num_actions, device=self.device)
        self.dof_vel = th.zeros(self.num_actions, device=self.device)
        self.tcp_pos = th.zeros(3, device=self.device)
        # self.tcp_vel = th.zeros(3, device=self.device)
        self.object_pos = th.zeros(3, device=self.device)
        self.target_pos = th.tensor(
            ENV_CFG["target_pos"], device=self.device, dtype=th.float32
        )

        self.actions = th.zeros(self.num_actions, device=self.device)
        self.last_actions = th.zeros(self.num_actions, device=self.device)
        self.last_dof_vel = th.zeros(self.num_actions, device=self.device)

        self.reward_functions = {
            "near": self._reward_near,
            "dist": self._reward_dist,
            "control": self._reward_control,
        }

        self.episode_sums = {key: 0.0 for key in self.reward_functions}

        self.reset()

    def step(self, action):
        action = np.clip(action, -ENV_CFG["clip_actions"], ENV_CFG["clip_actions"])
        self.actions.copy_(th.from_numpy(action).to(self.device))

        target_dof_pos = self.dof_pos + self.actions * ENV_CFG["action_scale"]
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        self.scene.step()
        self.episode_step += 1

        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)
        self.tcp_pos = self.robot.get_links_pos()[7, :]
        # self.tcp_vel = self.robot.get_links_vel()[7, :]
        self.object_pos = self.object.get_pos()

        dist_to_target = th.norm(self.target_pos - self.object_pos)
        success = bool((dist_to_target < self.target_threshold).item())

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        # if success:
        #     reward += 5
        reward = float(reward.item())

        self.episode_return += reward

        terminated = bool(success)
        truncated = self.episode_step >= self.max_episode_length

        info = {
            "success": success,
            "dist_to_target": dist_to_target.item(),
            "episode_step": self.episode_step,
            "is_truncated": truncated,
            "is_success": success,
        }

        for key, value in self.episode_sums.items():
            info[f"reward_{key}"] = value.item()

        if terminated or truncated:
            info["episode"] = {"r": float(reward), "l": self.episode_step}

        obs = (
            th.cat(
                [
                    self.dof_pos * self.obs_scales["dof_pos"],
                    self.dof_vel * self.obs_scales["dof_vel"],
                    self.tcp_pos,
                    # self.tcp_vel,
                    self.object_pos,
                    self.target_pos,
                ]
            )
            .cpu()
            .numpy()
        )

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            th.manual_seed(seed)

        self.dof_pos = self.default_dof_pos.clone()
        self.dof_vel = th.zeros_like(self.dof_vel)
        self.robot.set_dofs_position(
            position=self.dof_pos,
            dofs_idx_local=self.motor_dofs,
            zero_velocity=True,
        )
        self.robot.zero_all_dofs_velocity()

        self.tcp_pos = self.robot.get_links_pos()[7, :].float()
        # self.tcp_vel = self.robot.get_links_vel()[7, :].float()

        x_range = ENV_CFG["object_spawn_x"]
        y_range = ENV_CFG["object_spawn_y"]
        z_range = ENV_CFG["object_spawn_z"]

        rand_pos = th.tensor(
            [
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
                np.random.uniform(z_range[0], z_range[1]),
            ],
            device=self.device,
        )
        default_quat = th.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)

        self.object.set_pos(rand_pos, zero_velocity=True)
        self.object.set_quat(default_quat, zero_velocity=True)
        self.object_pos[:] = self.object.get_pos()

        self.actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.last_dof_vel[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}

        obs = (
            th.cat(
                [
                    self.dof_pos * self.obs_scales["dof_pos"],
                    self.dof_vel * self.obs_scales["dof_vel"],
                    self.tcp_pos,
                    # self.tcp_vel,
                    self.object_pos,
                    self.target_pos,
                ]
            )
            .cpu()
            .numpy()
        )

        return obs, {}

    def _reward_near(self):
        return th.norm(self.tcp_pos - self.object_pos)

    def _reward_dist(self):
        return th.norm(self.target_pos - self.object_pos)

    def _reward_control(self):
        return th.sum(self.actions**2)

    def render(self):
        pass
