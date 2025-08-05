import time
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from .ros_interface import ROSInterface


episode_length = 30
num_obs = 18
num_actions = 6
target_threshold = 0.02
clip_action = 1
obs_scales = {"dof_pos": 1.0, "dof_vel": 0.1}
action_scale = 0.1
reward_scales = {"dist": -1.0, "control": -0.1}
target_spawn_x = [-0.26, 0.26]
target_spawn_y = [0.36, 1.0]
target_spawn_z = [0.98, 1.78]


class AegisReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode=None,
        reward_type="dense",
        control_type="joints",
    ):
        super().__init__()

        self.robot = ROSInterface()
        self.num_obs = num_obs
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.obs_scales = obs_scales

        self.num_actions = num_actions
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.action_scale = action_scale

        self.reward_scales = reward_scales
        self.target_threshold = target_threshold

        self.episode_step = 0.0

        self.actions = torch.zeros(self.num_actions)
        self.target_pos = torch.zeros(3)
        self.dof_pos = torch.zeros(6)
        self.dof_vel = torch.zeros(6)
        self.tcp_pos = torch.zeros(3)

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

    def step(self, action):
        action = np.clip(action, -clip_action, clip_action)
        self.actions.copy_(torch.tensor(action, dtype=torch.float32))

        self.dof_pos = self.robot.get_joint_positions()
        delta = self.actions.clone().detach() * self.action_scale
        dof_pos_target = self.dof_pos + delta
        self.robot.control_dofs_position(dof_pos_target)
        self.tcp_pos = self.robot.get_tcp_position()

        self.episode_step += 1

        self.dist = torch.norm(self.tcp_pos - self.target_pos)
        success = bool((self.dist < self.target_threshold).item())

        reward = 0.0
        for name, func in self.reward_functions.items():
            r = func() * self.reward_scales[name]
            self.episode_sums[name] += r
            reward += r
        reward = float(reward.item())
        self.episode_return += reward

        current_time = time.time()
        elapsed_time = current_time - self.episode_start_time

        terminated = bool(success)
        truncated = elapsed_time >= episode_length

        info = self._get_info(reward, terminated, truncated, success)

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.robot.move_to_home()

        x_range = target_spawn_x
        y_range = target_spawn_y
        z_range = target_spawn_z

        self.target_pos = torch.tensor(
            [
                np.random.uniform(x_range[0], x_range[1]),
                np.random.uniform(y_range[0], y_range[1]),
                np.random.uniform(z_range[0], z_range[1]),
            ]
        )
        self.robot.publish_target_pos(self.target_pos)

        self.actions[:] = 0.0
        self.episode_step = 0
        self.episode_return = 0.0
        self.episode_sums = {k: 0.0 for k in self.reward_functions}
        self.tcp_pos = self.robot.get_tcp_position()
        self.dist = torch.norm(self.tcp_pos - self.target_pos)
        self.episode_start_time = time.time()

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        self.dof_pos = self.robot.get_joint_positions()
        self.dof_vel = self.robot.get_joint_velocities()
        self.tcp_pos = self.robot.get_tcp_position()
        return (
            torch.cat([self.dof_pos, self.dof_vel, self.tcp_pos, self.target_pos])
            .cpu()
            .numpy()
        )

    def _get_info(self, reward=0.0, terminated=False, truncated=False, success=False):
        info = {
            "success": success,
            "dist_to_target": self.dist.item(),
            "episode_step": self.episode_step,
            "is_truncated": truncated,
            "is_success": success,
        }

        for key, value in self.episode_sums.items():
            info[f"reward_{key}"] = float(value)

        if terminated or truncated:
            info["episode"] = {"r": float(reward), "l": self.episode_step}

        return info

    def _reward_dist(self):
        return self.dist

    def _reward_control(self):
        return torch.sum(self.actions**2)

    def render(self):
        pass
