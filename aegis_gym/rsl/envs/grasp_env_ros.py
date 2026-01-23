import math
from typing import Literal

import torch as th
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .manipulator_ros import ManipulatorROS


class GraspEnvROS(VecEnv):
    def __init__(
        self,
        env_cfg: dict,
        reward_cfg: dict,
        robot_cfg: dict,
        show_viewer: bool = False,
        device: th.device = th.device("cpu"),
    ) -> None:
        self.cfg = env_cfg
        self.device = device
        self.show_ciewer = show_viewer

        self._exctract_config()
        self.reward_scales = reward_cfg

        self.robot = ManipulatorROS(
            num_envs=self.num_envs,
            args=robot_cfg,
            device=self.device,
        )

        # TODO: an entity with get_pose() method
        self.object = None

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _exctract_config(self) -> None:
        self.num_envs = self.cfg["num_envs"]
        self.num_obs = self.cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = self.cfg["num_actions"]
        self.image_width = self.cfg["image_resolution"][0]
        self.image_height = self.cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = self.cfg["visualize_cell"]
        self.camera_setup: Literal["default", "scene_dual"] = self.cfg["camera_setup"]

        self.box_size = self.cfg["box_size"]
        self.table_size = self.cfg["table_size"]
        self.workbench_size = self.cfg["workbench_size"]

        # Probably not the wisest choice, it should be hardcoded
        # self.ctrl_dt = self.cfg["ctrl_dt"]
        self.ctrl_dt = 1.0 / 10.0  # 1/policy_f [s]

        self.sim_substeps = self.cfg["sim_substeps"]
        self.max_episode_length = math.ceil(self.cfg["episode_length_s"] / self.ctrl_dt)

        self.reward_scales = self.cfg["reward_scales"]
        self.action_scales = th.tensor(self.cfg["action_scales"], device=self.device)

    # Required by rsl_rl
    @property
    def unwrapped(self) -> "GraspEnvROS":
        return self

    # Required by rsl_rl
    @property
    def step_dt(self) -> float:
        return self.ctrl_dt

    # Required by rsl_rl
    @property
    def cfg(self) -> dict:
        return self.cfg

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=self.device, dtype=th.float32
            )

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, device=self.device, unit_length=0.5
        )

    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.num_envs,), device=self.device, dtype=th.float32
        )
        self.reset_buf = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        self.goal_pose = th.zeros(self.num_envs, 7, device=self.device)
        self.extras = dict()
        self.extras["observations"] = dict()

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        # reset robot
        self.robot.reset(envs_idx)

        # reset object
        # TODO - needs an object entity
        # num_reset = len(envs_idx)
        # random_x = th.rand(num_reset, device=self.device) * 0.22 + 0.36  # 0.36 – 0.58
        # random_y = (th.rand(num_reset, device=self.device) - 0.5) * 0.4  # -0.2 – 0.2
        # random_z = th.ones(num_reset, device=self.device) * (
        #     self.table_size[2] - self.workbench_size[2] + self.box_size[2] / 2
        # )
        # random_pos = th.stack([random_x, random_y, random_z], dim=-1)

        # # downward facing quaternion to align with the hand
        # q_downward = th.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(
        #     num_reset, 1
        # )
        # # randomly yaw the object
        # random_yaw = (
        #     th.rand(num_reset, device=self.device) * 2 * math.pi - math.pi
        # ) * 0.25
        # q_yaw = th.stack(
        #     [
        #         th.cos(random_yaw / 2),
        #         th.zeros(num_reset, device=self.device),
        #         th.zeros(num_reset, device=self.device),
        #         th.sin(random_yaw / 2),
        #     ],
        #     dim=-1,
        # )
        # goal_yaw = transform_quat_by_quat(q_yaw, q_downward)

        # self.goal_pose[envs_idx] = th.cat([random_pos, goal_yaw], dim=-1)
        # self.object.set_pos(random_pos, envs_idx=envs_idx)
        # self.object.set_quat(goal_yaw, envs_idx=envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self.cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        return self.get_observations(), self.extras

    def step(self, actions: th.Tensor) -> tuple[TensorDict, th.Tensor, th.Tensor, dict]:
        # update time
        self.episode_length_buf += 1

        # apply action based on task
        actions = actions * self.action_scales

        self.robot.apply_action(actions, open_gripper=True)
        self.robot.read_state()

        # check termination
        env_reset_idx = self.is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=self.device, dtype=th.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs = self.get_observations()
        dones = self.reset_buf
        return obs, reward, dones, self.extras

    # currently not in use
    def get_privileged_observations(self) -> None:
        return None

    def is_episode_complete(self) -> th.Tensor:
        time_out_buf = self.episode_length_buf > self.max_episode_length

        # check if the end-effector is in the valid position
        self.reset_buf = time_out_buf

        # fill time out buffer for reward/value bootstrapping
        time_out_idx = (time_out_buf).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = th.zeros_like(
            self.reset_buf, device=self.device, dtype=th.float32
        )
        self.extras["time_outs"][time_out_idx] = 1.0
        return self.reset_buf.nonzero(as_tuple=True)[0]

    def get_observations(self) -> TensorDict:
        ee_pos, ee_quat = (
            self.robot.ee_pose[:, :3],
            self.robot.ee_pose[:, 3:7],
        )
        # TODO get object entity
        # obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        obs_components = [
            # TODO apply object state space
            # ee_pos - obj_pos,  # 3D position difference
            ee_quat,  # current orientation (w, x, y, z)
            # obj_pos,  # goal position
            # obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return TensorDict({"policy": obs_tensor}, batch_size=[self.num_envs])

    def get_stereo_rgb_images(self, normalize: bool = True) -> th.Tensor:
        # Currently no IMAGES
        # TODO implement gRPC image transportation

        # rgb_left, _, _, _ = self.scene_left_cam.render(
        #     rgb=True, depth=False, segmentation=False, normal=False
        # )
        # rgb_right, _, _, _ = self.scene_right_cam.render(
        #     rgb=True, depth=False, segmentation=False, normal=False
        # )

        # # convert to the NCHW format
        # rgb_left = rgb_left.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)
        # rgb_right = rgb_right.permute(0, 3, 1, 2)[:, :3]  # shape (N, 3, H, W)

        # # normalize if requested
        # if normalize:
        #     rgb_left = th.clamp(rgb_left, min=0.0, max=255.0) / 255.0
        #     rgb_right = th.clamp(rgb_right, min=0.0, max=255.0) / 255.0

        # # concatenate left and right rgb images along channel dimension
        # stereo_rgb = th.cat([rgb_left, rgb_right], dim=1)
        # return stereo_rgb
        return th.zeros(3, dtype=th.float32, device=self.device)

    def get_observations_vis(self, normalize: bool = True) -> th.Tensor:
        raise NotImplementedError
        # match self.camera_setup:
        #     case "default":
        #         cams = [self.scene_cam, self.tool_left_cam, self.tool_right_cam]
        #     case "scene_dual":
        #         cams = [self.scene_left_cam, self.scene_right_cam]
        #     case _:
        #         raise ValueError(f"Unknown camera setup {self.camera_setup}")

        # rgb_list = []
        # for cam in cams:
        #     rgb, _, _, _ = cam.render(
        #         rgb=True, depth=False, segmentation=False, normal=False
        #     )
        #     rgb = rgb.permute(0, 3, 1, 2)[:, :3]
        #     if normalize:
        #         rgb = th.clamp(rgb, 0.0, 255.0) / 255.0
        #     rgb_list.append(rgb)

        # rgb_multi = th.cat(rgb_list, dim=1)
        # return rgb_multi

    def _reward_keypoints(self) -> th.Tensor:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        keypoints_offset = self.keypoints_offset
        object_offset = th.tensor(
            [0.0, 0.0, -0.08],
            device=self.device,
            dtype=th.float32,
        ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            ee_pos + object_offset,
            ee_quat,
            keypoints_offset,
        )
        # TODO object entity
        # object_pos_keypoints = self._to_world_frame(
        #     self.object.get_pos(), self.object.get_quat(), keypoints_offset
        # )
        # dist = th.norm(finger_pos_keypoints - object_pos_keypoints, p=2, dim=-1).sum(-1)
        dist = th.norm(finger_pos_keypoints, p=2, dim=-1).sum(-1)
        return th.exp(-dist)

    def _to_world_frame(
        self,
        position: th.Tensor,  # [N, 3]
        quaternion: th.Tensor,  # [N, 4]
        keypoints_offset: th.Tensor,  # [N, 7, 3]
    ) -> th.Tensor:
        world = th.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            # TODO
            # world[:, k] = position + transform_by_quat(
            #     keypoints_offset[:, k], quaternion
            # )
            world[:, k] = position
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

    def grasp_and_lift_demo(self) -> None:
        total_steps = 500
        grab_height = 0.08
        goal_pose = self.robot.ee_pose.clone()
        goal_pose[:, 2] -= grab_height
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
        reset_pose = th.tensor(
            [0.2, 0.0, 0.4, 0.0, 1.0, 0.0, 0.0], device=self.device
        ).repeat(self.num_envs, 1)
        for i in range(total_steps):
            if i < total_steps / 5:  # go down
                self.robot.go_to_goal(goal_pose, open_gripper=True)
            elif i < total_steps * 2 / 5:  # grasping
                self.robot.go_to_goal(goal_pose, open_gripper=False)
            elif i < total_steps * 3 / 5:  # lifting
                self.robot.go_to_goal(lift_pose, open_gripper=False)
            elif i < total_steps * 4 / 5:  # final
                self.robot.go_to_goal(final_pose, open_gripper=False)
            else:  # reset
                self.robot.go_to_goal(reset_pose, open_gripper=True)
            self.robot.read_state()
