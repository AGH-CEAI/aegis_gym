import math
from typing import Optional

import genesis as gs
import numpy as np
from tensordict import TensorDict
import torch as th
from genesis.vis.camera import Camera
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

from .manipulator import GenesisManipulator
from .base_env import BaseEnv, StepReturn, ResetReturn, Modality
from .objects import BaseBox, ObjectsFactory
from .plotjuggler_udp import PlotJugglerUDP

from config.types import ExpConfig, CameraPoseCfg, Control, CamerasSetup

# Further example
# https://github.com/isaac-sim/IsaacLab/blob/857da263c08fa78664e40ab957f996b22153d181/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py


class GraspEnv(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})

    def __init__(
        self,
        cfg: ExpConfig,
    ) -> None:
        super().__init__(
            scene=None, cfg_env=cfg.env_cfg
        )  # TODO(issue#128) introduce Scene abstraction
        self.device = cfg.get_device()

        self._observation_fns = {
            Modality.TCP_POSE: self._observe_tcp_pose,
            Modality.OBJECT_POSE: self._observe_object_pose,
            Modality.CAMERA_SCENE_RGB: self._observe_camera_scene,
            Modality.CAMERA_TOOL_LEFT_RGB: self._observe_camera_tool_left,
            Modality.CAMERA_TOOL_RIGHT_RGB: self._observe_camera_tool_right,
        }

        enable_plot_juggler = cfg.args.enable_plotjuggler
        if enable_plot_juggler:
            ip = "127.0.0.1"
            port = 9870
            self._joint_names = [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
            self._pj = PlotJugglerUDP(host=ip, port=port)
            print(f"[GraspEnv] Enabled UDP server for PlotJuggler at {ip}:{port}")
        self._enable_pj_logging = enable_plot_juggler

        self._cfg_env = cfg.env_cfg
        self._extract_config()
        self.device = cfg.get_device()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg_env.visualize_camera

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._cfg_env.num_actions
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = self._cfg_env.visualize_cell
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size
        self.box_size = self._cfg_env.box_size_default

        self.ctrl_dt = self._cfg_env.ctrl_dt
        self.policy_dt = self._cfg_env.policy_dt
        self.sim_substeps = int(
            math.ceil(self._cfg_env.policy_dt / self._cfg_env.ctrl_dt)
        )
        self.max_episode_length = int(
            math.ceil(self._cfg_env.episode_length_s / self.policy_dt)
        )

        self.max_linear_speed = self._cfg_env.action_max_linear_speed
        self.max_angular_speed = self._cfg_env.action_max_angular_speed

        self.reward_scales = self._cfg_env.reward_scales

    def get_policy_dt(self) -> float:
        return self.policy_dt

    def get_cfg_as_dict(self) -> dict:
        return self._cfg_env.as_dict()

    def get_num_envs(self) -> int:
        return self.num_envs

    def _observe_tcp_pose(self) -> th.Tensor:
        return self.robot.get_tcp_pose()

    def _observe_object_pose(self) -> th.Tensor:
        return self.object.get_pose()

    def _observe_camera_scene(self) -> th.Tensor:
        return self._render_rgb_camera("scene_cam")

    def _observe_camera_tool_left(self) -> th.Tensor:
        return self._render_rgb_camera("tool_left_cam")

    def _observe_camera_tool_right(self) -> th.Tensor:
        return self._render_rgb_camera("tool_right_cam")

    def _render_rgb_camera(self, camera: str) -> th.Tensor:
        rgb, _, _, _ = self._cameras[camera].render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb = rgb.permute(0, 3, 1, 2)[:, :3]  # (B, 3, H, W)
        rgb = th.clamp(rgb, 0.0, 255.0).div_(255.0)
        return rgb

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # Reset the robot
        self.robot.ctrl_gripper_open(envs_idx)
        self.robot.ctrl_go_to_home(envs_idx)

        # reset object
        num_reset = len(envs_idx)
        random_x = th.rand(num_reset, device=self.device) * 0.22 + 0.36  # 0.36 – 0.58
        random_y = (th.rand(num_reset, device=self.device) - 0.5) * 0.4  # -0.2 – 0.2
        random_z = th.ones(num_reset, device=self.device) * (
            self.table_size[2] - self.workbench_size[2] + self.box_size[2] / 2
        )
        random_pos = th.stack([random_x, random_y, random_z], dim=-1)

        # downward facing quaternion to align with the hand
        q_downward = th.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
            num_reset, 1
        )
        # randomly yaw the object
        random_yaw = (
            th.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        ) * 0.25
        q_yaw = th.stack(
            [
                th.cos(random_yaw / 2),
                th.zeros(num_reset, device=self.device),
                th.zeros(num_reset, device=self.device),
                th.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        goal_pose = th.cat([random_pos, goal_yaw], dim=-1)
        self.goal_pose[envs_idx] = goal_pose
        self.object.set_pose(pose=goal_pose, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg_env.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        if self._cfg_dr.enabled:
            self._setup_dr_pd_gains()
            self._randomize_camera_extrinsics(envs_idx)

    # TODO do something wit it
    def generate_object_poses(self, seed: int) -> th.Tensor:
        rng = th.Generator(device=self.device)
        rng.manual_seed(seed)

        random_x = (
            th.rand(self.num_envs, device=self.device, generator=rng) * 0.22 + 0.36
        )
        random_y = (
            th.rand(self.num_envs, device=self.device, generator=rng) - 0.5
        ) * 0.4
        random_z = th.ones(self.num_envs, device=self.device) * (
            self.table_size[2] - self.workbench_size[2] + self.box_size[2] / 2
        )
        random_yaw = (
            th.rand(self.num_envs, device=self.device, generator=rng) * 2 * math.pi
            - math.pi
        ) * 0.25

        q_downward = th.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        q_yaw = th.stack(
            [
                th.cos(random_yaw / 2),
                th.zeros(self.num_envs, device=self.device),
                th.zeros(self.num_envs, device=self.device),
                th.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        object_quat = transform_quat_by_quat(q_yaw, q_downward)
        object_pos = th.stack([random_x, random_y, random_z], dim=-1)

        return th.cat([object_pos, object_quat], dim=-1)

    def apply_object_poses(self, pose: th.Tensor) -> None:
        self.object.set_pose(pose=pose)
        self.goal_pose[:] = pose

    def _reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=gs.device))
        self._log_state_to_plot_juggler()
        return ResetReturn(self.get_observations(), self.extras)

    def _step(self, actions: th.Tensor) -> StepReturn:
        # Update time
        self.episode_length_buf += 1

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        # Applying real-world scaling (with optional noise)
        max_lin_speed, max_ang_speed = self._get_max_speed_coeefs()
        actions[:, :3] *= max_lin_speed
        actions[:, 3:] *= max_ang_speed

        self.robot.ctrl_apply_vel_action(actions, open_gripper=True)
        self.scene.step()

        self._log_state_to_plot_juggler()

        # check termination
        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs = self.get_observations()
        dones = self.reset_buf
        return StepReturn(obs, reward, dones, self.extras)

    def calib_run(
        self,
        joints_diff: Optional[th.Tensor] = None,
        cart_diff: Optional[th.Tensor] = None,
        steps: int = 100,
    ) -> None:
        idle_steps = 300  # int(0.4 * steps)
        print(f">>> Idling for {idle_steps} steps.")
        for _ in range(idle_steps):
            self.scene.step()
            self._log_state_to_plot_juggler()

        move_steps = int(steps)
        steps_per_action = int(
            250 / 10
        )  # Control Frequency divided by Policy Frequency
        # move_per_action = 0.106 / 1000 * steps_per_action # about 2.65 mm
        move_per_action = 0.196 / 500 * steps_per_action  # about 9,8mm
        steady_error_compensation_coeff = 1.03  # 1.087 #1.044 # 1.0472

        print(f">>> Moving to relative goal for {move_steps} steps.")
        if joints_diff is not None:
            self.robot.ctrl_apply_joints_diff_action(joints_diff)
        elif cart_diff is not None:
            from math import ceil

            print(f">>> Steps per action: {steps_per_action}.")
            print(f">>> Movement per action: {move_per_action} m.")
            actions_num = int(ceil(move_steps / steps_per_action))
            print(
                f">>> Assuming, that the goal will be reachable in: {actions_num} actions."
            )

            scaled_cart_diff = cart_diff / actions_num * steady_error_compensation_coeff
            print(f">>> Scaled down target: {scaled_cart_diff}")
            for action_id in range(actions_num):
                print(f">>> Applying action #{action_id + 1}")
                self.robot.ctrl_apply_vel_action(scaled_cart_diff, open_gripper=True)
                for _ in range(steps_per_action):
                    self.scene.step()
                    self._log_state_to_plot_juggler()

        print(f">>> Idling for {idle_steps} steps.")
        for _ in range(idle_steps):
            self.scene.step()
            self._log_state_to_plot_juggler()

