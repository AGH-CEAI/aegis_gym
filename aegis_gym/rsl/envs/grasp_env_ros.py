import math
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch as th
from genesis.utils.geom import transform_by_quat
from PIL import Image
from tensordict import TensorDict

from .manipulator import RosGrpcManipulator, CameraID
from .base_env import BaseEnv, StepReturn, ResetReturn


class Object:
    def __init__(
        self,
        size: th.Tensor,
        pos: th.Tensor,
        quat: th.Tensor,
        num_envs: int = 1,
        device: Optional[th.device] = None,
    ):
        self.device = device or th.device("cpu")
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


class GraspEnvROS(BaseEnv):
    def __init__(
        self,
        env_cfg: dict,
        robot_cfg: dict,
        disable_vision: bool = False,
        device: Optional[th.device] = None,
    ) -> None:
        super().__init__(scene=None)  # TODO: introduce Scene abstraction
        self.device = device or th.device("cpu")
        self._cfg = env_cfg
        self.disable_vision = disable_vision

        self._extract_config()
        print(
            f"[GraspEnvROS] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self.robot = RosGrpcManipulator(
            num_envs=self.num_envs,
            args=robot_cfg,
            disable_vision=self.disable_vision,
            device=self.device,
        )

        # This pose is already in the Genesi's world base
        world_box_pose = th.tensor(
            # TODO(issue#98) move setup into URDF-dataset in ClearML
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

        # TODO(issue#41) Unify the setup of the cameras
        # TODO(issue#121) Unify the grasp_env and grasp_env_ros cameras names
        match self.camera_setup:
            case "default":
                self._cameras = ["scene", "left", "right"]
            case "scene_dual":
                self._cameras = ["left", "right"]
            case _:
                raise ValueError(f"Unknown camera setup {self.camera_setup}")
        self._cameras_order = {
            "scene": 0,
            "left": 1,
            "right": 2,
        }

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

        self.last_step_ts: Optional[float] = None

    def get_policy_dt(self) -> float:
        return self.policy_dt

    def get_cfg_as_dict(self) -> dict:
        return self._cfg

    def get_num_envs(self) -> int:
        return self.num_envs

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
        self.robot.ctrl_gripper_open(envs_idx)
        self.robot.ctrl_go_to_home(envs_idx)
        self.object.reset(envs_idx)

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self.robot.read_state()
        return ResetReturn(self.get_observations(), self.extras)

    def step(self, actions: th.Tensor) -> StepReturn:
        if not self.last_step_ts:
            self.last_step_ts = time.perf_counter()

        # update time
        self.episode_length_buf += 1

        # Environment limitations
        actions = th.clamp(actions, min=-1.0, max=1.0)

        while time.perf_counter() - self.last_step_ts < self.policy_dt:
            time.sleep(0.0001)
        self.last_step_ts = time.perf_counter()

        self.robot.ctrl_apply_vel_action(actions)
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
        return StepReturn(obs, reward, dones, self.extras)

    def calib_run(
        self,
        joints_diff: Optional[th.Tensor] = None,
        cart_diff: Optional[th.tensor] = None,
        steps: int = 100,
    ) -> None:
        raise NotImplementedError

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
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pos, tcp_quat = (
            tcp_pose[:, :3],
            tcp_pose[:, 3:],
        )
        obj_pos, obj_quat = self.object.get_pos(), self.object.get_quat()

        obs_components = [
            tcp_pos - obj_pos,  # 3D position difference
            tcp_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return TensorDict({"policy": obs_tensor}, batch_size=[self.num_envs])

    def get_stereo_rgb_images(self, normalize: bool = True) -> th.Tensor:
        # TODO implement gRPC image transportation
        raise NotImplementedError

    def get_observations_vis(
        self,
        normalize: bool = True,
        swap_tool_cameras: bool = False,
        enable_vis_preview: bool = False,
        enable_record_obs: bool = False,
        record_dir: Optional[Path] = None,
    ) -> th.Tensor:
        # TODO(issue#41) Unify the camera setup
        rgb_list: list[th.Tensor] = [None] * len(self._cameras)

        for cam_name in self._cameras:
            cam_id = self._cameras_order[cam_name]
            if swap_tool_cameras:
                cam_name_tmp = {"left": "right", "right": "left"}.get(
                    cam_name, cam_name
                )
                cam_id = self._cameras_order[cam_name_tmp]

            rgb = self.robot.get_camera_image(CameraID.from_str(cam_name))
            if normalize:
                rgb = th.clamp(rgb, 0.0, 255.0).div_(255.0)
            rgb_list[cam_id] = rgb

        # TODO(issue#117) redesign the visualization preview
        # TODO(issue#121) unify the code between grasp_env and grasp_env_ros
        if enable_vis_preview or enable_record_obs:
            preview = self._create_vis_observation_preview(
                obs=rgb_list, normalize=normalize
            )

            if enable_vis_preview:
                cv2.imshow("Visual observation preview", preview)
                cv2.waitKey(1)

            if enable_record_obs:
                record_dir = (
                    record_dir or Path(tempfile.gettempdir()) / "aegis_vis_obs_obs"
                )
                record_dir.mkdir(parents=True, exist_ok=True)
                self._record_vis_observation(preview=preview, output_dir=record_dir)

        return th.cat(rgb_list, dim=1).float()

    def _create_vis_observation_preview(
        self, obs: list[th.Tensor], normalize: bool
    ) -> np.ndarray:

        opencv_images: list[np.ndarray] = [None] * len(obs)
        for cam_name in self._cameras:
            cam_id = self._cameras_order[cam_name]

            FIRST_IMG = 0
            img = obs[cam_id][FIRST_IMG].permute(1, 2, 0).cpu().numpy()  # NCHW -> HWC
            img = (img * 255).astype(np.uint8) if normalize else img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            height, width = img.shape[:2]
            max_side = 256
            scale = min(max_side / width, max_side / height)
            img = cv2.resize(
                img,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
            cv2.putText(
                img,
                str(cam_name),
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 255, 0),
                thickness=2,
            )
            opencv_images[cam_id] = img

        return np.hstack(opencv_images)

    def _record_vis_observation(self, preview: np.ndarray, output_dir: Path) -> None:
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        record_step = getattr(self, "_record_step", 0)
        fname = f"frame_{record_step:08d}.png"
        Image.fromarray(preview).save(output_dir / fname)

        self._record_step = record_step + 1

    def _reward_keypoints(self) -> th.Tensor:
        tcp_pose = self.robot.get_tcp_pose()
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        keypoints_offset = self.keypoints_offset
        object_offset = th.tensor(
            [0.0, 0.0, -0.08],
            device=self.device,
            dtype=th.float32,
        ).repeat(self.num_envs, 1)

        finger_pos_keypoints = self._to_world_frame(
            tcp_pos + object_offset,
            tcp_quat,
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
        self.robot.read_state()

        total_steps = 500
        grab_height = 0.08
        min_width = 0.005
        max_width = 0.04
        goal_pose = self.robot.get_tcp_pose().clone()
        goal_pose[:, 2] -= grab_height

        # lift pose (above the object)
        lift_height = 0.16
        lift_pose = goal_pose.clone()
        lift_pose[:, 2] += lift_height

        print("[GraspEnvROS] Proceeding with the GraspAndLift demo")
        step_1 = False
        step_2 = False
        step_3 = False
        step_4 = False
        success = False

        try:
            for i in range(total_steps):
                self.robot.read_state()
                if i < total_steps / 5:  # go down
                    if step_1:
                        continue
                    print("[GraspEnvROS][Demo] STEP 1: Going down to the grasp pose")
                    self.robot.ctrl_go_to_goal(goal_pose)
                    step_1 = True
                elif i < total_steps * 2 / 5:  # grasping
                    if step_2:
                        continue
                    print("[GraspEnvROS][Demo] STEP 2: Grasping")
                    self.robot.ctrl_gripper_close()
                    step_2 = True
                elif i < total_steps * 3 / 5:  # lifting
                    if step_3:
                        continue
                    print("[GraspEnvROS][Demo] STEP 3: Going up to the lift pose")
                    self.robot.ctrl_go_to_goal(lift_pose)
                    fingers_width = self.robot.get_gripper_width()
                    success = (fingers_width > min_width) and (
                        fingers_width < max_width
                    )
                    step_3 = True
                else:  # reset
                    if step_4:
                        continue
                    print("[GraspEnvROS][Demo] STEP 4: Going home")
                    self.robot.ctrl_go_to_home()
                    self.robot.ctrl_gripper_open()
                    step_4 = True
        except Exception as e:
            print(f"[GraspEnvROS][Demo] Caught an exception: {e}")
            success = 0.0
            print("[GraspEnvROS][Demo] Going home")
            self.robot.ctrl_go_to_home()
            self.robot.ctrl_gripper_open()

        print(f"[GraspEnvROS][Demo] Grasp success: {success}")
        return float(success)
