import math
from typing import Literal, Optional

import genesis as gs
import numpy as np
import torch as th
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

from .manipulator import Manipulator
from .plotjuggler_udp import PlotJugglerUDP

# Further example
# https://github.com/isaac-sim/IsaacLab/blob/857da263c08fa78664e40ab957f996b22153d181/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py


class GraspEnv(VecEnv):
    def __init__(
        self,
        env_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
        enable_plot_juggler: bool = False,
    ) -> None:
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

        self._cfg = env_cfg
        self.device = gs.device

        self._extract_config()
        print(
            f"[GraspEnv] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self._cameras: dict[str, gs.Camera] = {}
        self._setup_genesis_scene(self._cfg, robot_cfg, show_viewer)

        self.scene.build(
            n_envs=env_cfg["num_envs"],
            # env_spacing=(1.0, 1.0),
        )

        self.robot.set_pd_gains()
        self._attach_cameras()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
        self.show_cameras_gui = self._cfg["visualize_camera"]

        self.num_envs = self._cfg["num_envs"]
        self.num_obs = self._cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = self._cfg["num_actions"]
        self.image_width = self._cfg["image_resolution"][0]
        self.image_height = self._cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = self._cfg["visualize_cell"]
        self.camera_setup: Literal["default", "scene_dual"] = self._cfg["camera_setup"]
        self.table_size = self._cfg["table_size"]
        self.workbench_size = self._cfg["workbench_size"]
        self.box_size = self._cfg["box_sizes"]["default"]

        self.ctrl_dt = self._cfg["ctrl_dt"]
        self.policy_dt = self._cfg["policy_dt"]
        self.sim_substeps = int(
            math.ceil(self._cfg["policy_dt"] / self._cfg["ctrl_dt"])
        )
        self.max_episode_length = int(
            math.ceil(self._cfg["episode_length_s"] / self.policy_dt)
        )

        self.max_linear_speed = self._cfg["action_scaling"]["max_linear_speed"]
        self.max_angular_speed = self._cfg["action_scaling"]["max_angular_speed"]

        self.reward_scales = self._cfg["reward_scales"]

    def _setup_genesis_scene(
        self, env_cfg: dict, robot_cfg: dict, show_viewer: bool
    ) -> None:
        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.policy_dt,
                substeps=self.sim_substeps,
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.policy_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(self.num_envs, 10))),
            ),
            viewer_options=gs.options.ViewerOptions(
                # max_FPS=int(0.5 / self.ctrl_dt),
                max_FPS=int(60),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            renderer=gs.options.renderers.BatchRenderer(
                use_rasterizer=env_cfg["use_rasterizer"],
            ),
            show_viewer=show_viewer,
        )

        # == add ground ==
        plane_z = -0.82 if self.show_cell else 0.0
        self.scene.add_entity(gs.morphs.Plane(pos=(0, 0, plane_z)))

        # == add robot ==
        self.robot = Manipulator(
            num_envs=self.num_envs,
            scene=self.scene,
            args=robot_cfg,
            show_cell=self.show_cell,
            device=gs.device,
        )

        # == add table ==
        if self.show_cell:
            self.table = self.scene.add_entity(
                gs.morphs.Box(
                    size=self.table_size,
                    pos=(
                        self.table_size[0] / 2 + self.workbench_size[0] / 2,
                        0.0,
                        self.table_size[2] / 2 - self.workbench_size[2],
                    ),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(1.0, 0.96, 0.92)),
                material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
            )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=self.box_size,
                fixed=env_cfg["box_fixed"],
                collision=env_cfg["box_collision"],
            ),
            # material=gs.materials.Rigid(gravity_compensation=1),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
        )

        # == add cameras ==
        match self.camera_setup:
            case "default":
                self._add_camera(name="scene_cam", fov=40)
                self._add_camera(name="tool_left_cam", fov=30)
                self._add_camera(name="tool_right_cam", fov=30)
            case "scene_dual":
                self._add_camera(name="scene_left_cam", pos=(1.25, 0.3, 0.3), fov=60)
                self._add_camera(name="scene_right_cam", pos=(1.25, -0.3, 0.3), fov=60)

        if self.show_cameras_gui:
            self.record_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self.show_cameras_gui,
                debug=True,
            )

        # == add lighting ==
        self.scene.add_light(
            pos=(0.0, 0.0, 2.46),
            dir=(1.0, 1.0, -1.0),
            color=(1.0, 1.0, 1.0),
            intensity=0.6,
            directional=False,
            castshadow=True,
            cutoff=90.0,
        )

    # TODO(issue#41): Refactor camera handling to use a unified camera registry instead of dynamic attributes
    def _add_camera(
        self,
        name: str,
        pos: tuple = (0.0, 0.0, 0.0),
        fov: float = 40,  # deg
        lookat: tuple = (0.0, 0.0, 0.0),
        res: tuple = None,
    ):
        if res is None:
            res = (self.image_width, self.image_height)
        self._cameras[name] = self.scene.add_camera(
            res=res,
            pos=pos,
            lookat=lookat,
            fov=fov,
            GUI=self.show_cameras_gui,
        )

    def _attach_cameras(self):
        if self.camera_setup != "default":
            return

        scene_offset_T = np.array(
            [
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        tool_offset_T = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -0.03],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        cams_to_attach = [
            ("scene_cam", "cam_scene_rgb_camera_frame", scene_offset_T),
            ("tool_left_cam", "cam_tool_left", tool_offset_T),
            ("tool_right_cam", "cam_tool_right", tool_offset_T),
        ]

        for cam_name, link_name, offset in cams_to_attach:
            self._cameras[cam_name].attach(
                self.robot._robot_entity.get_link(link_name), offset
            )
            self._cameras[cam_name].move_to_attach()

    # Required by rsl_rl
    @property
    def unwrapped(self) -> "GraspEnv":
        return self

    # Required by rsl_rl
    @property
    def step_dt(self) -> float:
        return self.policy_dt

    # Required by rsl_rl
    @property
    def cfg(self) -> dict:
        return self._cfg

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt * self.sim_substeps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self.device, unit_length=0.5
        )

    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = th.zeros(self.num_envs, dtype=th.bool, device=gs.device)
        self.goal_pose = th.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

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

        self.goal_pose[envs_idx] = th.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def generate_box_poses(self, seed: int = 42) -> tuple[th.Tensor, th.Tensor]:
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
        box_quat = transform_quat_by_quat(q_yaw, q_downward)
        box_pos = th.stack([random_x, random_y, random_z], dim=-1)

        return box_pos, box_quat

    def apply_box_poses(self, box_pos: th.Tensor, box_quat: th.Tensor) -> None:
        self.object.set_pos(box_pos)
        self.object.set_quat(box_quat)
        self.goal_pose[:] = th.cat([box_pos, box_quat], dim=-1)

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=gs.device))
        self._log_state_to_plot_juggler()
        return self.get_observations(), self.extras

    def step(self, actions: th.Tensor) -> tuple[TensorDict, th.Tensor, th.Tensor, dict]:
        # Update time
        self.episode_length_buf += 1

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        # Applying real-world scaling
        actions[:, :3] *= self.max_linear_speed
        actions[:, 3:] *= self.max_angular_speed

        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()
        if self.show_cameras_gui:
            self.get_observations_vis()
        self._log_state_to_plot_juggler()

        # check termination
        env_reset_idx = self.is_episode_complete()
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
        return obs, reward, dones, self.extras

    def calib_run(
        self,
        joints_diff: Optional[th.Tensor] = None,
        cart_diff: Optional[th.tensor] = None,
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
            self.robot.apply_dof_rel_action(joints_diff)
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
                self.robot.apply_action(scaled_cart_diff, open_gripper=True)
                for _ in range(steps_per_action):
                    self.scene.step()
                    self._log_state_to_plot_juggler()

        print(f">>> Idling for {idle_steps} steps.")
        for _ in range(idle_steps):
            self.scene.step()
            self._log_state_to_plot_juggler()

    def get_privileged_observations(self) -> None:
        raise NotImplementedError

    def is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the end-effector is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> TensorDict:
        ee_pos, ee_quat = (
            self.robot.ee_pose[:, :3],
            self.robot.ee_pose[:, 3:7],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        obs_components = [
            ee_pos - obj_pos,  # 3D position difference
            ee_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return TensorDict({"policy": obs_tensor}, batch_size=[self.num_envs])

    def get_stereo_rgb_images(self, normalize: bool = True) -> th.Tensor:
        rgb_left, _, _, _ = self.scene_left_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_right, _, _, _ = self.scene_right_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )

        # convert to the NCHW format
        rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)
        rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)

        # normalize if requested
        if normalize:
            rgb_left = th.clamp(rgb_left, min=0.0, max=255.0) / 255.0
            rgb_right = th.clamp(rgb_right, min=0.0, max=255.0) / 255.0

        # concatenate left and right rgb images along channel dimension
        stereo_rgb = th.cat([rgb_left, rgb_right], dim=1)
        return stereo_rgb

    def get_observations_vis(self, normalize: bool = True) -> th.Tensor:
        cams = tuple(self._cameras.values())
        rgb_list = [None] * len(cams)

        for cam_id, cam in enumerate(cams):
            rgb, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )
            rgb = rgb.permute(0, 3, 1, 2)[:, :3]
            if normalize:
                rgb = th.clamp(rgb, 0.0, 255.0) / 255.0
            rgb_list[cam_id] = rgb

        return th.cat(rgb_list, dim=1)

    def _reward_keypoints(self) -> th.Tensor:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        keypoints_offset = self.keypoints_offset
        object_offset = th.tensor(
            [0.0, 0.0, -0.08],
            device=self.device,
            dtype=gs.tc_float,
        ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            ee_pos + object_offset,
            ee_quat,
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(
            self.object.get_pos(), self.object.get_quat(), keypoints_offset
        )
        dist = th.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        return th.exp(-dist)

    def _to_world_frame(
        self,
        position: th.Tensor,  # [N, 3]
        quaternion: th.Tensor,  # [N, 4]
        keypoints_offset: th.Tensor,  # [N, 7, 3]
    ) -> th.Tensor:
        world = th.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(
                keypoints_offset[:, k], quaternion
            )
        return world

    @staticmethod
    def get_keypoint_offsets(
        batch_size: int, device: str, unit_length: float = 0.5
    ) -> th.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            th.tensor(
                [
                    [0, 0, 0],  # origin
                    [-1.0, 0, 0],  # x-negative
                    [1.0, 0, 0],  # x-positive
                    [0, -1.0, 0],  # y-negative
                    [0, 1.0, 0],  # y-positive
                    [0, 0, -1.0],  # z-negative
                    [0, 0, 1.0],  # z-positive
                ],
                device=device,
                dtype=th.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    def grasp_and_lift_demo(self) -> float:
        total_steps = self.max_episode_length
        grab_height = 0.08
        goal_pose = self.robot.ee_pose.clone()
        goal_pose[:, 2] -= grab_height
        # lift pose (above the object)
        lift_height = 0.16
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        # final pose (above the table)
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        # reset pose (home pose)
        reset_pose = th.tensor(
            [0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)

        pos_threshold = 0.08
        hold_steps_required = self.max_episode_length / 10
        hold_counter = th.zeros(self.num_envs, device=self.device)

        for i in range(total_steps):
            if i < total_steps / 5:  # go down
                self.robot.go_to_goal(goal_pose, open_gripper=True)
            elif i < total_steps * 2 / 5:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps * 3 / 5:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 4 / 5:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
                obj_pos = self.object.get_pos()
                target_pos = final_pose[:, :3]

                dist = th.norm(obj_pos - target_pos, dim=-1)
                in_target = dist < pos_threshold

                hold_counter[in_target] += 1
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()

        success = hold_counter >= hold_steps_required
        success_rate = success.float().mean().item()

        return success_rate

    def _log_state_to_plot_juggler(self) -> None:
        if not self._enable_pj_logging:
            return

        data = {}
        robot = self.robot._robot_entity
        for name in self._joint_names:
            j = robot.get_joint(name=name)
            for idx in j.dofs_idx_local:
                # Query each DOF individually to get scalar values
                pos = robot.get_dofs_position([idx])
                vel = robot.get_dofs_velocity([idx])
                force = robot.get_dofs_force([idx])

                # Convert to float - handle both tensor and array shapes
                data[f"joint_states/{name}/position"] = float(pos.flatten()[0])
                data[f"joint_states/{name}/velocity"] = float(vel.flatten()[0])
                data[f"joint_states/{name}/effort"] = float(force.flatten()[0])

        all_link_positions = robot.get_links_pos()
        # all_link_quats = robot.get_links_quat()

        link_positions = all_link_positions[0]
        # link_quats = all_link_quats[0]

        ee_idx = -1  # Last link = end effector
        position = link_positions[ee_idx]

        data["ee/position/x"] = float(position[0])
        data["ee/position/y"] = float(position[1])
        data["ee/position/z"] = float(position[2])
        # TODO(issue#55) Enable orientation logging
        # data["ee/orientation/roll"] = float(roll)
        # data["ee/orientation/pitch"] = float(pitch)
        # data["ee/orientation/yaw"] = float(yaw)

        self._pj.send(data)
