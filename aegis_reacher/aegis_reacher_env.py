import os
import importlib.util
import time
import torch
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ament_index_python.packages import get_package_share_directory


episode_length = 20
num_obs = 18
num_actions = 6
target_threshold = 0.01
clip_action = 1
obs_scales = {"dof_pos": 1.0, "dof_vel": 0.1}
action_scale = 0.5
reward_scales = {"dist": -1.0, "control": -0.1}
target_spawn_x = [-0.3, 0.3]
target_spawn_y = [0.28, 0.68]
target_spawn_z = [0.9, 1.7]

director_pkg_path = get_package_share_directory("aegis_director")
robot_director_path = os.path.join(director_pkg_path, "aegis_director", "robot_director.py")

spec = importlib.util.spec_from_file_location("robot_director", robot_director_path)
robot_director = importlib.util.module_from_spec(spec)
spec.loader.exec_module(robot_director)

RobotDirector = robot_director.RobotDirector

class ROSInterface:
    def __init__(self):
        self.robot_director = RobotDirector(synchronous=True)
        self.joint_names = self.robot_director.joint_names
        self.dof_home = {
            name: value for name, value in zip(self.joint_names, [0.0, -1.57, 1.57, 0.0, 0.0, 0.0])
        }

    def get_joint_positions(self) -> torch.Tensor:
        js = self.robot_director.get_joint_states()
        return torch.tensor([js[name] for name in self.joint_names], dtype=torch.float32)

    def get_joint_velocities(self) -> torch.Tensor:
        jv = self.robot_director.get_joint_velocities()
        return torch.tensor([jv[name] for name in self.joint_names], dtype=torch.float32)

    def get_tcp_position(self) -> torch.Tensor:
        tcp = self.robot_director.get_tcp_pose()
        return torch.tensor(tcp["position"], dtype=torch.float32)

    def control_dofs_position(self, target_pos: torch.Tensor, max_vel: float = 0.3, max_accel: float = 0.3):
        joint_dict = {
            name: float(pos)
            for name, pos in zip(self.joint_names, target_pos)
        }
        self.robot_director.joint_move(joint_positions=joint_dict, max_vel=max_vel, max_accel=max_accel)

    def move_to_home(self, dof_pos):
        self.robot_director.joint_move(
            joint_positions=self.dof_home,
            max_vel=0.5,
            max_accel=0.5,
        )

    def __del__(self):
        pass


class AegisReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, device="cuda", render_mode=None):
        super().__init__()

        self.robot = ROSInterface()
        self.device = device

        self.num_obs = num_obs
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32)
        self.obs_scales = obs_scales
    
        self.num_actions = num_actions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32)
        self.action_scale = action_scale

        self.reward_scales = reward_scales
        self.target_threshold = target_threshold

        self.episode_step = 0.0
        self.episode_start_time = 0.0

        self.actions = torch.zeros(self.num_actions, device=self.device)
        self.target_pos = torch.zeros(3, device=self.device)

        self.reward_functions = {
            "dist": self._reward_dist,
            "control": self._reward_control,
        }

        self.episode_sums = {key: 0.0 for key in self.reward_functions}

        self.reset()

    def step(self, action):
        action = np.clip(action, -clip_action, clip_action)
        self.actions.copy_(torch.tensor(action, dtype=torch.float32, device=self.device))

        dof_pos = self.robot.get_joint_positions()
        delta = torch.tensor(self.actions, dtype=torch.float32) * self.action_scale
        dof_pos_target = dof_pos + delta
        self.robot.control_dofs_position(dof_pos_target)
        tcp_pos = self.robot.get_tcp_position()

        self.episode_step += 1

        self.dist = torch.norm(tcp_pos - self.target_pos)
        success = bool((self.dist < self.target_threshold).item())

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        # if success:
        #     reward += 5
        reward = float(reward.item())
        self.episode_return += reward

        current_time = time.time()
        elapsed_time = current_time - self.episode_start_time

        terminated = bool(success)
        truncated = elapsed_time >= episode_length

        info = {
            "success": success,
            "dist_to_target": self.dist.item(),
            "episode_step": self.episode_step,
            "is_truncated": truncated,
            "is_success": success,
        }

        for key, value in self.episode_sums.items():
            info[f"reward_{key}"] = value.item()

        if terminated or truncated:
            info["episode"] = {
                "r": float(reward),
                "l": self.episode_step
            }

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.robot.move_to_home()

        x_range = target_spawn_x
        y_range = target_spawn_y
        z_range = target_spawn_z

        self.target_pos = torch.tensor([
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
            np.random.uniform(z_range[0], z_range[1]),
        ], device=self.device)

        self.actions[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}

        self.episode_start_time = time.time()

        return self._get_obs(), {}
    
    def _get_obs(self):
        dof_pos = self.robot.get_joint_positions()
        dof_vel = self.robot.get_joint_velocities()
        tcp_pos = self.robot.get_tcp_position()
        return torch.cat([dof_pos, dof_vel, tcp_pos, self.target_pos]).numpy()

    def _reward_dist(self):
        return self.dist

    def _reward_control(self):
        return torch.sum(self.actions ** 2)

    def render(self):
        pass
