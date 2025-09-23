# Original implementation by Jakub Płachno (sivral) 2025
# Major refactor by Maciej Aleksandrowicz (macmacal) 2025
from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from ..scene import (
    SceneDirectorType,
    SceneDirectorInterface,
    RobotCommanderInterface,
    get_scene_director,
    EntityType,
    Target,
    Box,
)
from .env_types import EnvControlType, EnvObservationType, EnvRewardType, EnvRenderMode

ENV_CFG = {
    "max_episode_length": 1000,
    "num_obs": 21,
    "target_pos": [-0.1, 0.76, 0.84],
    "target_threshold": 0.04,
    "object_spawn_x": [-0.36, -0.24],
    "object_spawn_y": [0.34, 0.66],
    "object_spawn_z": [0.84, 0.85],
    "clip_action": 1,
    "action_scale": 0.1,
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


class AegisPusherEnv(gym.Env):
    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str = EnvRenderMode.NONE.name,
        observation_type: str = EnvObservationType.STATE.name,
        control_type: str = EnvControlType.JOINTS.name,
        reward_type: str = EnvRewardType.DENSE.name,
        scene_type: SceneDirectorType = SceneDirectorType.MOCK,
        device: str = "cuda",
        cfg: dict = ENV_CFG,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.render_mode = EnvRenderMode(render_mode)
        self.observation_type = EnvObservationType(observation_type)
        self.control_type = EnvControlType(control_type)
        self.reward_type = EnvRewardType(reward_type)

        self.num_actions = None
        if self.control_type == EnvControlType.JOINTS:
            self.num_actions = 6
        if self.control_type == EnvControlType.CARTESIAN_POSITION:
            self.num_actions = 3

        # TODO(issue#7) Rconsider episode length units unifcation for ROS and simulation
        self.max_episode_length = cfg["max_episode_length"]
        self.num_obs = cfg["num_obs"]
        self.target_threshold = cfg["target_threshold"]
        self.clip_action = cfg["clip_action"]
        self.obs_scales = cfg["obs_scales"]
        self.action_scale = cfg["action_scale"]
        self.reward_scales = cfg["reward_scales"]

        self.scene: SceneDirectorInterface = get_scene_director(scene_type)
        self.target: Target = self.scene.add_entity(EntityType.TARGET)
        self.object: Box = self.scene.add_entity(EntityType.BOX)
        self.target.create()
        self.object.create()

        self.scene.build()
        self.robot: RobotCommanderInterface = self.scene.get_robot_commander()

        self.observation_space: spaces.Space[th.Tensor] = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space: spaces.Space[th.Tensor] = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        self.episode_step = 0.0
        self.actions = th.zeros(self.num_actions, device=self.device)
        # self.last_actions = th.zeros(self.num_actions, device=self.device)
        # self.last_dof_vel = th.zeros(self.num_actions, device=self.device)
        self.dof_pos = th.zeros(6, device=self.device)
        self.dof_vel = th.zeros(6, device=self.device)
        self.tcp_pos = th.zeros(3, device=self.device)
        # self.tcp_vel = th.zeros(3, device=self.device)
        self.object_pos = th.zeros(3, device=self.device)
        self.target_pos = th.tensor(
            self.cfg["target_pos"], device=self.device, dtype=th.float32
        )

        self.reward_functions = {
            "near": self._reward_near,
            "dist": self._reward_dist,
            "control": self._reward_control,
        }
        self.episode_sums = {key: 0.0 for key in self.reward_functions}
        self.reset()

    def step(
        self, action: Union[th.Tensor, np.ndarray]
    ) -> tuple[th.Tensor, float, bool, bool, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = th.from_numpy(action)
        action = action.to(self.device)
        action = th.clamp(action, -self.clip_action, self.clip_action)
        self.actions = action.clone()
        delta = self.actions * self.action_scale

        if self.control_type == EnvControlType.JOINTS:
            dof_pos_target = self.dof_pos + delta
            self.robot.control_dofs_position(dof_pos_target)
        elif self.control_type == EnvControlType.CARTESIAN_POSITION:
            tcp_pos_target = self.tcp_pos + delta
            tcp_ori = self.robot.get_tcp_orientation()
            self.robot.control_tcp_position(
                target_pos=tcp_pos_target, target_ori=tcp_ori
            )

        self.scene.step()
        self.episode_step += 1

        obs = self._get_obs()

        self.dist_to_target = th.norm(self.target_pos - self.object_pos)
        success = bool(self.dist_to_target < self.target_threshold)

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        # if success:
        #     reward += 5
        reward = float(reward.item())

        self.episode_return += reward

        # TODO(issue#10) introduce timeouts in ROS and simulations
        # truncated = elapsed_time >= self.max_episode_length_s
        terminated = bool(success)
        truncated = self.episode_step >= self.max_episode_length

        info = self._get_info(reward, terminated, truncated, success)

        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[th.Tensor, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            th.manual_seed(seed)

        self.robot.move_to_home()

        x_range = self.cfg["object_spawn_x"]
        y_range = self.cfg["object_spawn_y"]
        z_range = self.cfg["object_spawn_z"]

        rand_pose = th.tensor(
            [
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
                np.random.uniform(z_range[0], z_range[1]),
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            device=self.device,
        )

        self.object.set_pose(rand_pose)
        obs = self._get_obs()
        self.dist_to_target = th.norm(self.tcp_pos - self.target_pos)

        self.actions[:] = 0.0
        # self.last_actions[:] = 0.0
        # self.last_dof_vel[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}

        return obs, {}

    def _get_obs(self) -> th.Tensor:
        self.dof_pos = self.robot.get_joint_positions()
        self.dof_vel = self.robot.get_joint_velocities()
        self.tcp_pos = self.robot.get_tcp_position()
        # self.tcp_vel = self.robot.get_tcp_velocity()
        self.object_pos = self.object.get_pose()[:3].clone()
        return (
            th.cat(
                [
                    self.dof_pos,
                    self.dof_vel,
                    self.tcp_pos,
                    self.object_pos,
                    self.target_pos,
                ]
            )
            .clone()
            .detach()
        )

    def _get_info(
        self,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        success: bool = False,
    ) -> dict[str, Any]:
        info = {
            "success": success,
            "dist_to_target": float(self.dist_to_target),
            "episode_step": self.episode_step,
            "is_truncated": truncated,
            "is_success": success,
        }

        for key, value in self.episode_sums.items():
            info[f"reward_{key}"] = float(value)

        if terminated or truncated:
            info["episode"] = {"r": float(reward), "l": self.episode_step}

        return info

    def _reward_near(self) -> th.Tensor:
        return th.norm(self.tcp_pos - self.object_pos)

    def _reward_dist(self) -> th.Tensor:
        return th.norm(self.target_pos - self.object_pos)

    def _reward_control(self) -> th.Tensor:
        return th.sum(self.actions**2)

    def render(self) -> None:
        print("AegisPusher Render not implemented yet.")
        pass
