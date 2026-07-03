import math
import time
from typing import Optional

import torch as th
from genesis.utils.geom import transform_by_quat

from .manipulator import RosGrpcManipulator, CameraID
from .base_env import BaseEnv, StepReturn, ResetReturn, Modality
from .objects import BaseBox, ObjectsFactory
from .scene import BaseScene

from config import ExpConfig
from config.types import Control, CamerasSetup


class GraspEnvROS(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})

    def __init__(self, cfg: ExpConfig, scene: Optional[BaseScene] = None) -> None:
        super().__init__(scene=scene)  # TODO(issue#128) introduce Scene abstraction

        self._observation_fns = {
            Modality.TCP_POSE: self._observe_tcp_pose,
            Modality.OBJECT_POSE: self._observe_object_pose,
            Modality.CAMERA_SCENE_RGB: self._observe_camera_scene,
            Modality.CAMERA_TOOL_LEFT_RGB: self._observe_camera_tool_left,
            Modality.CAMERA_TOOL_RIGHT_RGB: self._observe_camera_tool_right,
        }

        env_cfg = cfg.env_cfg
        self.device = cfg.get_device()

        self._cfg_env = env_cfg
        self.disable_vision = cfg.args.disable_vision

        self._extract_config()
        print(
            f"[GraspEnvROS] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self.robot = RosGrpcManipulator(
            num_envs=self.num_envs,
            scene=self._scene,
            robot_cfg=cfg.robot_cfg,
            policy_dt=cfg.env_cfg.policy_dt,
            disable_vision=self.disable_vision,
            device=self.device,
        )

        # This pose is already in the Genesi's world base
        # world_box_pose = th.tensor(
        #     # TODO(issue#98) move setup into URDF-dataset in ClearML
        #     # [0.631, 0.028, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0],
        #     # [0.557, 0.012, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0],
        #     [0.576, 0.245, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0],
        #     device=self.device,
        # )
        # world_box_pose[2] += 0.00  # m
        self.box_pose = (0.576, 0.245, self.box_size[2] / 2 + 0.02, 0.0, 1.0, 0.0, 0.0)

        self.object: BaseBox = ObjectsFactory.create_box(
            scene=self._scene, ctrl=Control.ROS, device=self.device
        )
        self.object.create(dims=self.box_size, pose=self.box_pose)

        # TODO(issue#41) Unify the setup of the cameras
        # TODO(issue#121) Unify the grasp_env and grasp_env_ros cameras names
        match self.cameras_setup:
            case CamerasSetup.DEFAULT:
                self._cameras = ["scene", "left", "right"]
            case CamerasSetup.SCENE_DUAL:
                self._cameras = ["left", "right"]

        self._cameras_order = {
            "scene": 0,
            "left": 1,
            "right": 2,
        }

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def _extract_config(self) -> None:
        self.num_envs = self._cfg_env.num_envs
        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._cfg_env.num_actions
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size
        self.box_size = tuple(self._cfg_env.box_size_default)

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

        self.last_step_ts: Optional[float] = None

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
        return self._read_rgb_camera(CameraID.SCENE_CAMERA)

    def _observe_camera_tool_left(self) -> th.Tensor:
        return self._read_rgb_camera(CameraID.TOOL_LEFT)

    def _observe_camera_tool_right(self) -> th.Tensor:
        return self._read_rgb_camera(CameraID.TOOL_RIGHT)

    def _read_rgb_camera(self, camera: CameraID) -> th.Tensor:
        rgb = self.robot.get_camera_image(camera_id=camera)
        rgb = th.clamp(rgb, 0.0, 255.0).div_(255.0)
        return rgb

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=self.device, dtype=th.float32
            )

        self.keypoints_offset = self.get_keypoint_offsets(
            batch_size=self.num_envs, unit_length=0.5
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
        self.object.set_pose(pose=th.tensor(self.box_pose, device=self.device))

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg_env.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self.robot.read_state()
        return ResetReturn(self.get_agent_observations(), self.extras)

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
        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=self.device, dtype=th.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        # get observations and fill extras
        obs = self.get_agent_observations()
        dones = self.reset_buf
        return StepReturn(obs, reward, dones, self.extras)

    def calib_run(
        self,
        joints_diff: Optional[th.Tensor] = None,
        cart_diff: Optional[th.Tensor] = None,
        steps: int = 100,
    ) -> None:
        raise NotImplementedError

    def _is_episode_complete(self) -> th.Tensor:
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

    def _agent_observations_mapping(self) -> th.Tensor:
        obs = self.get_observations()
        tcp_pose = obs[Modality.TCP_POSE]
        tcp_pos, tcp_quat = (
            tcp_pose[:, :3],
            tcp_pose[:, 3:],
        )
        obj_pose = obs[Modality.OBJECT_POSE]
        obj_pos, obj_quat = obj_pose[:, :3], obj_pose[:, 3:]

        obs_components = [
            tcp_pos - obj_pos,  # 3D position difference
            tcp_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        # TODO checkout if the self.extras was actually needed
        return th.cat(obs_components, dim=-1)

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

        obj_pose = self.object.get_pose()
        obj_pos, obj_quat = obj_pose[:, :3], obj_pose[:, 3:]
        object_pos_keypoints = self._to_world_frame(obj_pos, obj_quat, keypoints_offset)
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

    def get_keypoint_offsets(
        self, batch_size: int, unit_length: float = 0.5
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
                device=self.device,
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
