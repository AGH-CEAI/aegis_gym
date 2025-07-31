import time
import torch
import numpy as np
import rclpy
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from visualization_msgs.msg import Marker
from rclpy.clock import Clock
from std_msgs.msg import ColorRGBA
from aegis_director.robot_director import RobotDirector


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


class ROSInterface:
    _instance: Optional["ROSInterface"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ROSInterface, cls).__new__(cls)
        return cls._instance

    def __init__(self, device="cuda"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        rclpy.init()
        self.device = device
        self.robot_director = RobotDirector(synchronous=True)
        joint_state = self.robot_director._get_joint_states()
        self.joint_names = list(joint_state.name)[1:]
        self.dof_home = {
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -2.09,
            "elbow_joint": 2.09,
            "wrist_1_joint": -1.57,
            "wrist_2_joint": -1.57,
            "wrist_3_joint": 0.0,
            "robotiq_hande_left_finger_joint": 0.025,
        }
        self.marker_node = rclpy.create_node("marker_publisher")
        self.target_pub = self.marker_node.create_publisher(
            Marker, "/target_marker", 10
        )
        self._initialized = True

    def get_joint_positions(self) -> torch.Tensor:
        jp = self.robot_director.get_joint_positions()
        return torch.tensor(
            [jp[name] for name in self.joint_names], dtype=torch.float32
        ).to(self.device)

    def get_joint_velocities(self) -> torch.Tensor:
        jv = self.robot_director.get_joint_velocities()
        return torch.tensor(
            [jv[name] for name in self.joint_names], dtype=torch.float32
        ).to(self.device)

    def get_tcp_position(self) -> torch.Tensor:
        tcp = self.robot_director.get_tcp_pose()
        return torch.tensor(tcp["position"], dtype=torch.float32).to(self.device)

    def control_dofs_position(
        self, target_pos: torch.Tensor, max_vel: float = 0.3, max_accel: float = 0.3
    ):
        joint_dict = {
            name: float(pos) for name, pos in zip(self.joint_names, target_pos)
        }
        self.robot_director.joint_move(
            joint_positions=joint_dict, max_vel=max_vel, max_accel=max_accel
        )

    def move_to_home(self):
        self.robot_director.joint_move(
            joint_positions=self.dof_home,
            max_vel=0.5,
            max_accel=0.5,
        )

    def publish_target_pos(self, pos):
        msg = Marker()
        msg.header.frame_id = "world"
        msg.header.stamp = Clock().now().to_msg()

        msg.ns = "target"
        msg.id = 0
        msg.type = Marker.SPHERE
        msg.action = Marker.ADD

        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.w = 1.0

        msg.scale.x = 0.04
        msg.scale.y = 0.04
        msg.scale.z = 0.04

        msg.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        self.target_pub.publish(msg)

    def shutdown(self):
        rclpy.shutdown()

    def __del__(self):
        self.shutdown()


class AegisReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        device="cuda",
        render_mode=None,
        reward_type="dense",
        control_type="joints",
    ):
        super().__init__()

        self.robot = ROSInterface()
        self.device = device

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

        self.actions = torch.zeros(self.num_actions, device=self.device)
        self.target_pos = torch.zeros(3, device=self.device)
        self.dof_pos = torch.zeros(6, device=self.device)
        self.dof_vel = torch.zeros(6, device=self.device)
        self.tcp_pos = torch.zeros(3, device=self.device)

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
        self.actions.copy_(
            torch.tensor(action, dtype=torch.float32, device=self.device)
        )

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
        # if success:
        #     reward += 5
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
            ],
            device=self.device,
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
