import math
import torch

import genesis as gs
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

from manipulator import Manipulator


class GraspEnv:
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
    ) -> None:
        self.num_envs = env_cfg["num_envs"]
        self.num_obs = env_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.image_width = env_cfg["image_resolution"][0]
        self.image_height = env_cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = env_cfg["visualize_cell"]
        self.device = gs.device

        self.ctrl_dt = env_cfg["ctrl_dt"]
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.ctrl_dt)

        self.env_cfg = env_cfg
        self.reward_scales = reward_cfg
        self.action_scales = torch.tensor(env_cfg["action_scales"], device=self.device)

        self._init_scene(env_cfg, robot_cfg, show_viewer)
        self.scene.build(n_envs=env_cfg["num_envs"])

        self.robot.set_pd_gains()
        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _init_scene(self, env_cfg: dict, robot_cfg: dict, show_viewer: bool) -> None:
        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(env_cfg["num_envs"], 10)))
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.ctrl_dt),
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
                    size=env_cfg["table_size"],
                    pos=(
                        env_cfg["table_size"][0] / 2 + env_cfg["workbench_size"][0] / 2,
                        0.0,
                        env_cfg["table_size"][2] / 2 - env_cfg["workbench_size"][2],
                    ),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(0.5, 0.5, 0.5)),
                material=gs.materials.Rigid(friction=0.6, coup_friction=0.6),
            )

        # == add object ==
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                size=env_cfg["box_size"],
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
        if self.env_cfg["visualize_camera"]:
            self.vis_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self.env_cfg["visualize_camera"],
                debug=True,
            )

        # == add stereo camera ==
        self.left_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(1.25, 0.3, 0.3),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=self.env_cfg["visualize_camera"],
        )
        self.right_cam = self.scene.add_camera(
            res=(self.image_width, self.image_height),
            pos=(1.25, -0.3, 0.3),
            lookat=(0.0, 0.0, 0.0),
            fov=60,
            GUI=self.env_cfg["visualize_camera"],
        )

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self.device, unit_length=0.5
        )

    def _init_buffers(self) -> None:
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=gs.device)
        self.goal_pose = torch.zeros(self.num_envs, 7, device=gs.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: torch.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object
        num_reset = len(envs_idx)
        random_x = (
            torch.rand(num_reset, device=self.device) * 0.22 + 0.36
        )  # 0.36 – 0.58
        random_y = (
            torch.rand(num_reset, device=self.device) - 0.5
        ) * 0.5  # -0.25 – 0.25
        random_z = torch.ones(num_reset, device=self.device) * 0.025  # 0.15 – 0.15
        random_pos = torch.stack([random_x, random_y, random_z], dim=-1)

        # downward facing quaternion to align with the hand
        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
            num_reset, 1
        )
        # randomly yaw the object
        random_yaw = (
            torch.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        ) * 0.25
        q_yaw = torch.stack(
            [
                torch.cos(random_yaw / 2),
                torch.zeros(num_reset, device=self.device),
                torch.zeros(num_reset, device=self.device),
                torch.sin(random_yaw / 2),
            ],
            dim=-1,
        )
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        self.goal_pose[envs_idx] = torch.cat([random_pos, goal_yaw], dim=-1)
        self.object.set_pos(random_pos, envs_idx=envs_idx)
        self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        obs, self.extras = self.get_observations()
        return obs, self.extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # update time
        self.episode_length_buf += 1

        # apply action based on task
        actions = actions * self.action_scales

        self.robot.apply_action(actions, open_gripper=True)
        self.scene.step()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs, self.extras = self.get_observations()

        return obs, reward, self.reset_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None

    def is_episode_complete(self) -> torch.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the end-effector is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # current end-effector pose
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
        obs_tensor = torch.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor, self.extras

    def get_stereo_rgb_images(self, normalize: bool = True) -> torch.Tensor:
        rgb_left, _, _, _ = self.left_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )
        rgb_right, _, _, _ = self.right_cam.render(
            rgb=True, depth=False, segmentation=False, normal=False
        )

        # convert to the NCHW format
        rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)
        rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)

        # normalize if requested
        if normalize:
            rgb_left = torch.clamp(rgb_left, min=0.0, max=255.0) / 255.0
            rgb_right = torch.clamp(rgb_right, min=0.0, max=255.0) / 255.0

        # concatenate left and right rgb images along channel dimension
        stereo_rgb = torch.cat([rgb_left, rgb_right], dim=1)
        return stereo_rgb

    def _reward_keypoints(self) -> torch.Tensor:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        keypoints_offset = self.keypoints_offset
        # there is a offset between the finger tip and the finger base frame
        # finger_tip_z_offset = torch.tensor(
        #     [0.0, 0.0, -0.06],
        #     device=self.device,
        #     dtype=gs.tc_float,
        # ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            ee_pos,  # + finger_tip_z_offset,
            ee_quat,
            keypoints_offset,
        )
        object_pos_keypoints = self._to_world_frame(
            self.object.get_pos(), self.object.get_quat(), keypoints_offset
        )
        dist = torch.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(
            -1
        )
        return torch.exp(-dist)

    def _to_world_frame(
        self,
        position: torch.Tensor,  # [N, 3]
        quaternion: torch.Tensor,  # [N, 4]
        keypoints_offset: torch.Tensor,  # [N, 7, 3]
    ) -> torch.Tensor:
        world = torch.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(
                keypoints_offset[:, k], quaternion
            )
        return world

    @staticmethod
    def get_keypoint_offsets(
        batch_size: int, device: str, unit_length: float = 0.5
    ) -> torch.Tensor:
        """
        Get uniformly-spaced keypoints along a line of unit length, centered at body center.
        """
        keypoint_offsets = (
            torch.tensor(
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
                dtype=torch.float32,
            )
            * unit_length
        )
        return keypoint_offsets.unsqueeze(0).repeat(batch_size, 1, 1)

    def grasp_and_lift_demo(self) -> None:
        total_steps = 500
        goal_pose = self.robot.ee_pose.clone()
        # lift pose (above the object)
        lift_height = 0.3
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        # final pose (above the table)
        final_pose = goal_pose.clone()
        final_pose[:, 0] = 0.3
        final_pose[:, 1] = 0.0
        final_pose[:, 2] = 0.4
        # reset pose (home pose)
        reset_pose = torch.tensor(
            [0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)
        for i in range(total_steps):
            if i < total_steps / 4:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps / 2:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 3 / 4:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.scene.step()
