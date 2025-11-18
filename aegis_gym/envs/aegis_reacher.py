# Original implementation by Jakub Płachno (sivral) 2025
# Major refactor by Maciej Aleksandrowicz (macmacal) 2025
import time
from typing import Optional, Any, Union

import cv2
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

from ..scene import (
    SceneDirectorType,
    SceneDirectorInterface,
    RobotCommanderInterface,
    get_scene_director,
    EntityType,
    Target,
)
from .env_types import EnvControlType, EnvObservationType, EnvRewardType, EnvRenderMode

ENV_CFG = {
    "max_episode_length": 1000,
    "target_threshold": 0.02,
    "target_spawn_x": [-0.26, 0.26],
    "target_spawn_y": [0.36, 1.0],
    "target_spawn_z": [0.98, 1.78],
    "clip_action": 1,
    "action_scale": 0.1,
    "obs_scales": {"dof_pos": 1.0, "dof_vel": 0.1},
    "reward_scales": {"dist": -1.0, "control": -0.1},
}


class AegisReacherEnv(gym.Env):
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

        # TODO(issue#7) Rconsider episode length units unifcation for ROS and simulation
        self.max_episode_length = cfg["max_episode_length"]
        self.target_threshold = cfg["target_threshold"]
        self.clip_action = cfg["clip_action"]
        self.obs_scales = cfg["obs_scales"]
        self.action_scale = cfg["action_scale"]
        self.reward_scales = cfg["reward_scales"]
        self.target_spawn_x = cfg["target_spawn_x"]
        self.target_spawn_y = cfg["target_spawn_y"]
        self.target_spawn_z = cfg["target_spawn_z"]

        # TODO(issue#35) Remove temporal storage of scene_type used to fork JOINTS_SERVO calculatioans
        self.scene_type = scene_type

        enable_scene_camera = self.observation_type == EnvObservationType.MULTIMODAL
        self.scene: SceneDirectorInterface = get_scene_director(
            scene_type, enable_scene_camera
        )
        self.target: Target = self.scene.add_entity(EntityType.TARGET)
        self.target.create()

        self.scene.build()
        self.robot: RobotCommanderInterface = self.scene.get_robot_commander()

        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()

        self.episode_step = 0.0
        self.actions = th.zeros(
            self.action_space.shape, dtype=th.float32, device=self.device
        )
        self.target_pos = th.zeros(3, dtype=th.float32, device=self.device)
        self.dof_pos = th.zeros(6, dtype=th.float32, device=self.device)
        self.dof_vel = th.zeros(6, dtype=th.float32, device=self.device)
        self.tcp_pos = th.zeros(3, dtype=th.float32, device=self.device)
        self.base_pos = th.zeros(3, dtype=th.float32, device=self.device)

        self.reward_functions = {
            "dist": self._reward_dist,
            "control": self._reward_control,
        }
        self.episode_sums = {key: 0.0 for key in self.reward_functions}
        self.reset()

    def _make_observation_space(self) -> spaces.Space[th.Tensor]:
        match self.observation_type:
            case EnvObservationType.STATE:
                return spaces.Box(
                    low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32
                )
            case EnvObservationType.MULTIMODAL:
                return spaces.Dict(
                    {
                        "state": spaces.Box(
                            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
                        ),
                        "vision": spaces.Box(
                            low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
                        ),
                    }
                )
            case _:
                raise ValueError(
                    f"Unsupported observation type: {self.observation_type}"
                )

    def _make_action_space(self) -> spaces.Space[th.Tensor]:
        match self.control_type:
            case EnvControlType.JOINTS | EnvControlType.JOINTS_SERVO:
                return spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
            case (
                EnvControlType.CARTESIAN_POSITION
                | EnvControlType.CARTESIAN_POSITION_SERVO
            ):
                return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
            case _:
                raise ValueError(f"Unsupported control type: {self.control_type}")

    def step(
        self, action: Union[th.Tensor, np.ndarray]
    ) -> tuple[th.Tensor, float, bool, bool, dict[str, Any]]:
        if isinstance(action, np.ndarray):
            action = th.from_numpy(action)
        action = action.to(self.device)
        action = th.clamp(action, -self.clip_action, self.clip_action)
        self.actions = action.clone()
        delta = self.actions * self.action_scale

        match self.control_type:
            case EnvControlType.JOINTS:
                dof_pos_target = self.dof_pos + delta
                self.robot.control_dofs_position(dof_pos_target)
            case EnvControlType.JOINTS_SERVO:
                # TODO(issue#35) This should be speed in rad/s and it should be calibrated both for real and simulation purposes. Check the servo frequency in aegis_ros;s ur_servo.yaml configuration file.
                dof_pos_target = self.dof_pos + delta
                if (
                    self.scene_type == SceneDirectorType.ROS
                ):  # TODO(issue#35) remove condition
                    dof_pos_target = delta / self.action_scale * 0.5
                self.robot.control_dofs_position_servo(target_pos=dof_pos_target)
            case EnvControlType.CARTESIAN_POSITION:
                tcp_pos_target = self.tcp_pos + delta
                self.robot.control_tcp_position(target_pos=tcp_pos_target)
            case EnvControlType.CARTESIAN_POSITION_SERVO:
                # TODO(issue#35): similar joints case, this control should be given in m/s and it should be calibrated. For some reason the current implementation works with values only from 0.2 to 1.0 (the latter makes sense, that the maximum in the MoveiT2 Servo implementation, no foggiest idea what is the deal with 0.2).
                tcp_pos_target = self.tcp_pos + delta
                if (
                    self.scene_type == SceneDirectorType.ROS
                ):  # TODO(issue#35) remove condition
                    servo_delta = delta * 10.0 + 0.2
                    tcp_pos_target = th.clamp(
                        servo_delta, -self.clip_action, self.clip_action
                    )
                self.robot.control_tcp_position_servo(target_pos=tcp_pos_target)
            case _:
                raise ValueError(f"Unsupported control type: {self.control_type}")

        self.scene.step()
        self.episode_step += 1

        obs = self._get_obs()

        self.dist_to_target = th.norm(self.tcp_pos - self.target_pos)
        success = bool(self.dist_to_target < self.target_threshold)

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        reward = float(reward)

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
        self.base_pos = self.robot.get_base_position()

        self.target.set_pose(
            th.tensor(
                [
                    np.random.uniform(self.target_spawn_x[0], self.target_spawn_x[1]),
                    np.random.uniform(self.target_spawn_y[0], self.target_spawn_y[1]),
                    np.random.uniform(self.target_spawn_z[0], self.target_spawn_z[1]),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                device=self.device,
                dtype=th.float32,
            )
        )
        self.target_pos = self.target.get_pose()[:3]

        obs = self._get_obs()
        self.dist_to_target = th.norm(self.tcp_pos - self.target_pos)

        self.actions[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}
        self.episode_start_time = time.time()

        return obs, {}

    # TODO(issue#30) Fix return type
    def _get_obs(self) -> th.Tensor:
        self.dof_pos = self.robot.get_joint_positions()
        self.dof_vel = self.robot.get_joint_velocities()
        self.tcp_pos = self.robot.get_tcp_position()

        # Normalizing Cartesian positions w.r.t. the robot's base
        tcp_pos_rel = self.tcp_pos - self.base_pos
        target_pos_rel = self.target_pos - self.base_pos

        match self.observation_type:
            case EnvObservationType.STATE:
                return (
                    th.cat([self.dof_pos, self.dof_vel, tcp_pos_rel, target_pos_rel])
                    .clone()
                    .detach()
                )
            case EnvObservationType.MULTIMODAL:
                state_obs = (
                    th.cat([self.dof_pos, self.dof_vel, tcp_pos_rel]).clone().detach()
                )

                rgb, depth, seg, normal = self.scene.camera.render(
                    depth=False, segmentation=False, normal=False
                )
                vision_obs = self._process_rgb(rgb)

                return {"state": state_obs, "vision": vision_obs}
            case _:
                raise ValueError(
                    f"Unsupported observation type: {self.observation_type}"
                )

    def _process_rgb(self, rgb: np.ndarray) -> th.Tensor:
        rgb_proc = cv2.resize(
            np.ascontiguousarray(np.flipud(rgb)),
            (128, 128),
            interpolation=cv2.INTER_AREA,
        )
        return th.tensor(rgb_proc, dtype=th.float32, device=self.device) / 255.0

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

    def _reward_dist(self) -> float:
        return float(self.dist_to_target)

    def _reward_control(self) -> float:
        return float(th.sum(self.actions**2))

    def render(self) -> None:
        print("AegisReacher Render not implemented yet.")
        pass
