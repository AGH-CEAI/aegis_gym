import math
import time
from typing import Literal, Optional

import torch as th
from rsl_rl.env import VecEnv
from tensordict import TensorDict
from genesis.utils.geom import transform_by_quat

from .manipulator_ros import ManipulatorROS


class Object:
    def __init__(
        self,
        size: th.Tensor,
        pos: th.Tensor,
        quat: th.Tensor,
        num_envs: int = 1,
        device: th.device = th.device("cpu"),
    ):
        self.device = device
        self.num_envs = num_envs
        self.size = size

        self.pos = pos.repeat(self.num_envs, 1)
        self.quat = quat.repeat(self.num_envs, 1)
        self._init_pos = self.pos.clone()
        self._init_quat = self.quat.clone()

    def reset(self, envs_idx: int) -> None:
        self.pos = self._init_pos.clone()
        self.quat = self._init_quat.clone()

    def get_pos(self) -> th.Tensor:
        return self.pos

    def get_quat(self) -> th.Tensor:
        return self.quat

    def set_pos(self, pos: th.Tensor, envs_idx: int) -> None:
        self.pos = pos

    def set_quat(self, quat: th.Tensor, envs_idx: int) -> None:
        self.quat = quat


class GraspEnvROS(VecEnv):
    def __init__(
        self,
        env_cfg: dict,
        robot_cfg: dict,
        device: th.device = th.device("cpu"),
    ) -> None:
        self._cfg = env_cfg
        self.device = device

        self._extract_config()
        print(
            f"[GraspEnvROS] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self.robot = ManipulatorROS(
            num_envs=self.num_envs,
            args=robot_cfg,
            device=self.device,
        )

        # This pose is already in the Genesi's world base
        world_box_pose = th.tensor(
            # [0.631, 0.028, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0],
            # [0.557, 0.012, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0],
            [0.576, 0.245, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0],
            device=self.device,
        )
        world_box_pose[2] += 0.00  # m

        self.box_position = world_box_pose[:3]
        self.box_grasp_orientation = world_box_pose[3:]
        self.object = Object(
            size=self.box_size,
            pos=self.box_position,
            quat=self.box_grasp_orientation,
            num_envs=self.num_envs,
            device=self.device,
        )

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
        self.num_envs = self._cfg["num_envs"]
        self.num_obs = self._cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = self._cfg["num_actions"]
        self.image_width = self._cfg["image_resolution"][0]
        self.image_height = self._cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.camera_setup: Literal["default", "scene_dual"] = self._cfg["camera_setup"]
        self.table_size = self._cfg["table_size"]
        self.workbench_size = self._cfg["workbench_size"]
        self.box_size = self._cfg["box_size"]

        self.ctrl_dt = self._cfg["ctrl_dt"]
        self.policy_dt = self._cfg["policy_dt"]
        self.sim_substeps = int(
            math.ceil(self._cfg["policy_dt"] / self._cfg["ctrl_dt"])
        )
        self.max_episode_length = int(
            math.ceil(self._cfg["episode_length_s"] / self.policy_dt)
        )

        self.emperical_speed_coeff = self._cfg["emperical_speed_coeff"]
        self.emperical_speed_coeff_inv = 1 / self.emperical_speed_coeff
        self.max_linear_speed = self._cfg["max_linear_speed"]
        self.max_angular_speed = self._cfg["max_angular_speed"]

        self.reward_scales = self._cfg["reward_scales"]

        self.last_step_ts: Optional[float] = None

    # Required by rsl_rl
    @property
    def unwrapped(self) -> "GraspEnvROS":
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
        self.object.reset(envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self.robot.read_state()
        return self.get_observations(), self.extras

    def step(self, actions: th.Tensor) -> tuple[TensorDict, th.Tensor, th.Tensor, dict]:
        if not self.last_step_ts:
            self.last_step_ts = time.perf_counter()

        # update time
        self.episode_length_buf += 1

        # Agent related scaling
        actions *= self.emperical_speed_coeff_inv

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        while time.perf_counter() - self.last_step_ts < self.policy_dt:
            time.sleep(0.0001)
        self.last_step_ts = time.perf_counter()

        self.robot.apply_action(actions)
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

    def get_privileged_observations(self) -> None:
        raise NotImplementedError

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
        ee_pose = self.robot.ee_pose
        ee_pos, ee_quat = (
            ee_pose[:, :3],
            ee_pose[:, 3:7],
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
        # TODO implement gRPC image transportation
        raise NotImplementedError

    def get_observations_vis(self, normalize: bool = True) -> th.Tensor:
        # TODO implement gRPC image transportation
        raise NotImplementedError

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

    def grasp_and_lift_demo(self) -> None:
        self.robot.read_state()
        print(f"[GraspEnvROS] DEBUG: GraspAndLift ee_pose: {self.robot.ee_pose}")

        total_steps = 500
        grab_height = 0.08
        goal_pose = self.robot.ee_pose.clone()
        goal_pose[:, 2] -= grab_height
        print(f"[GraspEnvROS] DEBUG: GraspAndLift goal_pose: {goal_pose}")

        # lift pose (above the object)
        lift_height = 0.2
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height
        print(f"[GraspEnvROS] DEBUG: GraspAndLift lift_pose: {lift_pose}")

        print("[GraspEnvROS] Proceeding with the GraspAndLift demo")
        step_1 = False
        step_2 = False
        step_3 = False
        step_4 = False

        for i in range(total_steps):
            self.robot.read_state()
            if i < total_steps / 5:  # go down
                if step_1:
                    continue
                print("[GraspEnvROS] GOING DOWN TO THE GRASP POSE")
                self.robot.go_to_goal(goal_pose)
                step_1 = True
            elif i < total_steps * 2 / 5:  # grasping
                if step_2:
                    continue
                print("[GraspEnvROS] GRASPING")
                self.robot.gripper_close()
                step_2 = True
            elif i < total_steps * 3 / 5:  # lifting
                if step_3:
                    continue
                print("GOING UP TO LIFT POSE")
                self.robot.go_to_goal(lift_pose)
                step_3 = True
            else:  # reset
                if step_4:
                    continue
                print("GOING HOME")
                self.robot._move_to_home()
                self.robot.gripper_open()
                step_4 = True
