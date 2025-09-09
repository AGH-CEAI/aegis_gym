import os
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from aegis_gym.robot import RobotCommanderType

from aegis_gym.sim.genesis.genesis_commander import GenesisCommander


class AegisReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_type: str = "dense",
        control_type: str = "joints",
    ) -> None:
        super().__init__()

        self.episode_length = 30
        self.num_obs = 18
        self.num_actions = 6
        self.target_threshold = 0.02
        self.clip_action = 1
        self.obs_scales = {"dof_pos": 1.0, "dof_vel": 0.1}
        self.action_scale = 0.1
        self.reward_scales = {"dist": -1.0, "control": -0.1}
        self.target_spawn_x = [-0.26, 0.26]
        self.target_spawn_y = [0.36, 1.0]
        self.target_spawn_z = [0.98, 1.78]

        ros_interface = RobotCommanderType.REAL
        if os.environ.get("PYTEST_CURRENT_TEST") is not None:
            print("\n> Deteceted pytest env, using mock ROS interface")
            ros_interface = RobotCommanderType.MOCK
        # self.robot = get_ros_interface(ros_interface)
        print(f"Unused {ros_interface}")
        self.robot = GenesisCommander()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )

        self.episode_step = 0.0
        self.actions = np.zeros(self.num_actions, dtype=np.float32)
        self.target_pos = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(6, dtype=np.float32)
        self.dof_vel = np.zeros(6, dtype=np.float32)
        self.tcp_pos = np.zeros(3, dtype=np.float32)

        self.reward_functions = {
            "dist": self._reward_dist,
            "control": self._reward_control,
        }
        self.episode_sums = {key: 0.0 for key in self.reward_functions}

        assert (
            render_mode is None
            or render_mode in AegisReacherEnv.metadata["render_modes"]
        )
        self.render_mode = render_mode

        self.reset()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -self.clip_action, self.clip_action)
        self.actions = np.array(action, dtype=np.float32)

        self.dof_pos = self.robot.get_joint_positions()
        delta = self.actions * self.action_scale
        dof_pos_target = self.dof_pos + delta
        self.robot.control_dofs_position(dof_pos_target)
        self.robot.step()  # for simulation
        self.tcp_pos = self.robot.get_tcp_position()

        self.episode_step += 1
        self.dist = np.linalg.norm(self.tcp_pos - self.target_pos)
        success = bool(self.dist < self.target_threshold)

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        reward = float(reward)
        self.episode_return += reward

        current_time = time.time()
        elapsed_time = current_time - self.episode_start_time

        terminated = bool(success)
        truncated = elapsed_time >= self.episode_length
        info = self._get_info(reward, terminated, truncated, success)

        return self._get_obs(), reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        self.robot.move_to_home()

        self.target_pos = np.array(
            [
                np.random.uniform(self.target_spawn_x[0], self.target_spawn_x[1]),
                np.random.uniform(self.target_spawn_y[0], self.target_spawn_y[1]),
                np.random.uniform(self.target_spawn_z[0], self.target_spawn_z[1]),
            ],
            dtype=np.float32,
        )
        self.robot.publish_target_pos(self.target_pos)

        self.actions[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}
        self.tcp_pos = self.robot.get_tcp_position()
        self.dist = np.linalg.norm(self.tcp_pos - self.target_pos)
        self.episode_start_time = time.time()

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        self.dof_pos = self.robot.get_joint_positions()
        self.dof_vel = self.robot.get_joint_velocities()
        self.tcp_pos = self.robot.get_tcp_position()
        return np.concatenate(
            [self.dof_pos, self.dof_vel, self.tcp_pos, self.target_pos]
        )

    def _get_info(
        self,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        success: bool = False,
    ) -> Dict[str, Any]:
        info = {
            "success": success,
            "dist_to_target": self.dist,
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
        return self.dist

    def _reward_control(self) -> float:
        return np.sum(self.actions**2)

    def render(self) -> None:
        pass
