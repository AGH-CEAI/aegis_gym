import math

import torch as th
from tensordict import TensorDict

from aegis_gym.aux import transform_by_quat, transform_quat_by_quat
from aegis_gym.config.types import CameraName, ExpConfig
from aegis_gym.envs.base_env import BaseEnv, Modality, ResetReturn, StepReturn
from aegis_gym.envs.manipulator import BaseManipulator
from aegis_gym.envs.objects import BaseBox, ObjectProperties, ObjectType

from .scene import BaseScene


class ReacherEnv(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})

    def __init__(self, scene: BaseScene, cfg: ExpConfig):
        super().__init__(scene=scene, cfg=cfg)
        self._extract_config()
        self._observation_fns = {
            Modality.TCP_POSE: self._observe_tcp_pose,
            Modality.OBJECT_POSE: self._observe_object_pose,
            Modality.CAMERA_SCENE_RGB: self._observe_camera_scene,
            Modality.CAMERA_TOOL_LEFT_RGB: self._observe_camera_tool_left,
            Modality.CAMERA_TOOL_RIGHT_RGB: self._observe_camera_tool_right,
        }

        self._setup_scene(cfg=cfg)
        self._scene.build()
        self.manipulator: BaseManipulator = self._scene.get_manipulator()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    def get_cfg_as_dict(self) -> dict:
        return self._cfg_env.as_dict()

    def get_num_envs(self) -> int:
        return self.num_envs

    def _observe_tcp_pose(self) -> th.Tensor:
        return self.manipulator.get_tcp_pose()

    def _observe_object_pose(self) -> th.Tensor:
        return self.object.get_pose()

    def _observe_camera_scene(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_SCENE)

    def _observe_camera_tool_left(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_TOOL_LEFT)

    def _observe_camera_tool_right(self) -> th.Tensor:
        return self.manipulator.get_camera_image(camera=CameraName.CAMERA_TOOL_RIGHT)

    def _extract_config(self) -> None:
        # TODO(issue##117) redesign the whole camera preview system
        self.show_cameras_gui = self._cfg_env.visualize_camera
        self.show_cell = self._cfg_env.visualize_cell

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._cfg_env.num_actions
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size
        self.box_size = self._cfg_env.box_size_default

        self.ctrl_dt = self._cfg_env.ctrl_dt
        self.policy_dt = self._cfg_env.policy_dt
        self.sim_substeps = math.ceil(self._cfg_env.policy_dt / self._cfg_env.ctrl_dt)
        self.max_episode_length = math.ceil(
            self._cfg_env.episode_length_s / self.policy_dt
        )

        self.max_linear_speed = self._cfg_env.action_max_linear_speed
        self.max_angular_speed = self._cfg_env.action_max_angular_speed

        self.reward_scales = self._cfg_env.reward_scales

    def _setup_scene(self, cfg: ExpConfig) -> None:
        self._scene.add_manipulator(cfg=cfg.robot_cfg)
        p = ObjectProperties(
            dims=tuple(self.box_size),
            pose=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            collision=True,
            fixed=False,
            color=(0.8, 0.0, 0.0),
        )
        self.object: BaseBox = self._scene.add_entity(
            entity=ObjectType.BOX, properties=p
        )

    def _init_reward_functions(self) -> None:
        # TODO(issue#141) simplify creation of the rewards_functions registry
        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.ctrl_dt * self.sim_substeps
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = th.zeros(
                (self.num_envs,), device=self.device, dtype=th.float32
            )

        self.keypoints_offset = self._get_keypoint_offsets(
            batch_size=self.num_envs, unit_length=0.5
        )

    def _get_keypoint_offsets(
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

    def _init_buffers(self) -> None:
        self.episode_length_buf = th.zeros(
            (self.num_envs,), device=self.device, dtype=th.float32
        )
        self.reset_buf = th.zeros(self.num_envs, dtype=th.bool, device=self.device)
        self.goal_pose = th.zeros(self.num_envs, 7, device=self.device)
        self.extras = {}
        self.extras["observations"] = {}

    def _reset(self) -> ResetReturn:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=self.device))
        self._scene.update_state()
        return ResetReturn(self.get_observations(), self.extras)

    def reset_idx(self, envs_idx: th.Tensor) -> None:
        if len(envs_idx) == 0:
            return
        self.episode_length_buf[envs_idx] = 0

        self.manipulator.ctrl_gripper_open(envs_idx)
        self.manipulator.ctrl_go_to_home(envs_idx)

        obj_pose = self._get_random_object_pose(envs_idx=envs_idx)
        self.object.set_pose(pose=obj_pose, envs_idx=envs_idx)
        self.goal_pose[envs_idx] = obj_pose

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums:
            self.extras["episode"]["rew_" + key] = (
                th.mean(self.episode_sums[key][envs_idx]).item()
                / self._cfg_env.episode_length_s
            )
            self.episode_sums[key][envs_idx] = 0.0

        if not self._cfg_dr.enabled:
            return
        for rt in self._scene.get_available_randomizations():
            self._scene.randomize_domain(rand_type=rt, env_idx=envs_idx)

    def _get_random_object_pose(self, envs_idx: th.Tensor) -> th.Tensor:
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
        return goal_pose

    def _step(self, actions: th.Tensor) -> StepReturn:
        self.episode_length_buf += 1

        actions = th.clamp(actions, min=-1.0, max=1.0)

        self._scene.pre_step()
        self.manipulator.ctrl_apply_vel_action(actions, open_gripper=True)
        self._scene.step()

        # check env termination (Sets the self.reset_buf)
        env_reset_idx = self._is_episode_complete()
        if len(env_reset_idx) > 0:
            self.reset_idx(env_reset_idx)

        # compute reward based on task
        reward = th.zeros_like(self.reset_buf, device=self.device, dtype=th.float32)
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            reward += rew
            self.episode_sums[name] += rew

        obs = self.get_observations()
        dones = self.reset_buf
        return StepReturn(obs, reward, dones, self.extras)

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

    def _build_agent_observations(self, obs: TensorDict) -> th.Tensor:
        tcp_pose = obs[Modality.TCP_POSE]
        tcp_pos, tcp_quat = tcp_pose[:, :3], tcp_pose[:, 3:]
        obj_pose = obs[Modality.OBJECT_POSE]
        obj_pos, obj_quat = obj_pose[:, :3], obj_pose[:, 3:]

        obs_components = [
            tcp_pos - obj_pos,  # 3D position difference
            tcp_quat,  # current orientation (w, x, y, z)
            obj_pos,  # goal position
            obj_quat,  # goal orientation (w, x, y, z)
        ]
        obs_tensor = th.cat(obs_components, dim=-1)
        self.extras["observations"]["critic"] = obs_tensor
        return obs_tensor

    def _reward_keypoints(self) -> th.Tensor:
        tcp_pose = self.manipulator.get_tcp_pose()
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
        # TODO(issue#140) check this genesis implementation

        # N, K, _ = keypoints_offset.shape
        #
        # v_flat = keypoints_offset.reshape(N * K, 3)
        # quat_flat = quaternion[:, None].expand(N, K, 4).reshape(N * K, 4)
        # rotated = transform_by_quat(v_flat, quat_flat)
        # rotated = rotated.reshape(N, K, 3)
        # world = position[:, None, :] + rotated
        # return world
        world = th.zeros_like(keypoints_offset)
        for k in range(keypoints_offset.shape[1]):
            world[:, k] = position + transform_by_quat(
                keypoints_offset[:, k], quaternion
            )
        return world
