import asyncio
import math

import numpy as np
import torch as th
from tensordict import TensorDict
from rsl_rl.env import VecEnv
import genesis as gs
from genesis.utils.geom import (
    transform_by_quat,
    transform_quat_by_quat,
)

try:
    from aegis_grpc_client import AegisRobotClient
except ImportError:
    print(
        "Failed to import aegis_grpc_client. "
        "Double check if you have installed the `aegis_grpc_client` and `proto_aegis_grpc` packages."
    )
    raise

from ..scene import (
    SceneDirectorType,
)
from ..ros import RobotCommanderROS
from ..sim.genesis.robot_commander_genesis import RobotCommanderSimGenesis
from ..rsl.manipulator import Manipulator
from .env_types import EnvControlType, EnvObservationType, EnvRewardType, EnvRenderMode

# Further example
# https://github.com/isaac-sim/IsaacLab/blob/857da263c08fa78664e40ab957f996b22153d181/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py


ENV_CFG = {
    "num_envs": 10,
    "num_obs": 14,
    "num_actions": 6,
    "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
    "episode_length_s": 3.0,
    "ctrl_dt": 0.01,
    "sim_substeps": 2,  # 2 or 32
    "box_size": [0.03, 0.08, 0.06],
    "table_size": [0.55, 0.84, 0.82],
    "workbench_size": [0.64, 1.0, 0.806],
    "box_collision": False,
    "box_fixed": True,
    "image_resolution": (64, 64),
    "use_rasterizer": False,
    "visualize_camera": False,
    "visualize_cell": False,
    "camera_setup": "default",  # options: default, scene_dual
    "reward_scales": {
        "keypoints": 1.0,
    },
    "robot_cfg": {
        "ee_link_name": "robotiq_hande_end",
        "gripper_link_names": [
            "robotiq_hande_left_finger",
            "robotiq_hande_right_finger",
        ],
        "default_arm_dof": [0.0, -2.09, 2.09, -1.57, -1.57, 0.0],
        "default_gripper_dof": [0.025, 0.025],
        "ik_method": "dls_ik",
    },
}


class GraspEnv(VecEnv):
    def __del__(self) -> None:
        if not self.scene_type == SceneDirectorType.ROS:
            return
        if self._robot_client.is_connected:
            asyncio.run(self._robot_client.disconnect())

    def __init__(
        self,
        render_mode: str = EnvRenderMode.NONE.name,
        observation_type: str = EnvObservationType.STATE.name,
        control_type: str = EnvControlType.JOINTS.name,
        reward_type: str = EnvRewardType.DENSE.name,
        scene_type: SceneDirectorType = SceneDirectorType.MOCK,
        device: str = "cuda",
        cfg: dict = ENV_CFG,
    ) -> None:
        self.scene_type = scene_type

        self.num_envs = cfg["num_envs"]
        self.num_obs = cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = cfg["num_actions"]
        self.image_width = cfg["image_resolution"][0]
        self.image_height = cfg["image_resolution"][1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.show_cell = cfg["visualize_cell"]
        self.camera_setup = cfg["camera_setup"]
        self.device = gs.device

        self.ctrl_dt = cfg["ctrl_dt"]
        self.sim_substeps = cfg["sim_substeps"]
        self.max_episode_length = math.ceil(cfg["episode_length_s"] / self.ctrl_dt)

        self.cfg = cfg
        self.reward_scales = cfg["reward_scales"]
        self.action_scales = th.tensor(cfg["action_scales"], device=self.device)

        show_viewer = False
        if render_mode == EnvRenderMode.HUMAN.name:
            show_viewer = True
        self._init_scene(cfg, cfg["robot_cfg"], show_viewer)
        self.scene.build(n_envs=cfg["num_envs"])

        self.robot.set_pd_gains()
        self._attach_cameras()

        self._init_reward_functions()
        self._init_buffers()
        self.reset()

    # Required by rsl_rl
    @property
    def unwrapped(self) -> "GraspEnv":
        return self

    # Required by rsl_rl
    @property
    def step_dt(self) -> float:
        return self.ctrl_dt

    # Required by rsl_rl
    @property
    def cfg(self) -> dict:
        return self.cfg

    def _init_scene(self, env_cfg: dict, robot_cfg: dict, show_viewer: bool) -> None:
        if self.scene_type == SceneDirectorType.MOCK:
            return
        if self.scene_type == SceneDirectorType.ROS:
            self._robot_client = AegisRobotClient(server_address="127.0.0.1:50051")
            asyncio.run(self._robot_client.connect())
            self.robot_comm = RobotCommanderROS(self._robot_client)
            return

        # SceneDirectorType.SIM_GENESIS
        self._robot_client = None
        # == setup scene ==
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.ctrl_dt, substeps=self.sim_substeps
            ),
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
        self.robot_comm = RobotCommanderSimGenesis(
            self.robot,
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

        # == add cameras ==
        match self.camera_setup:
            case "default":
                self._add_camera(name="scene_cam", fov=40)
                self._add_camera(name="tool_left_cam", fov=30)
                self._add_camera(name="tool_right_cam", fov=30)
            case "scene_dual":
                self._add_camera(name="scene_left_cam", pos=(1.25, 0.3, 0.3), fov=60)
                self._add_camera(name="scene_right_cam", pos=(1.25, -0.3, 0.3), fov=60)

        if self.cfg["visualize_camera"]:
            self.record_cam = self.scene.add_camera(
                res=(1280, 720),
                pos=(1.5, 0.0, 0.2),
                lookat=(0.0, 0.0, 0.2),
                fov=60,
                GUI=self.cfg["visualize_camera"],
                debug=True,
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
        setattr(
            self,
            name,
            self.scene.add_camera(
                res=res,
                pos=pos,
                lookat=lookat,
                fov=fov,
                GUI=self.cfg["visualize_camera"],
            ),
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
            cam = getattr(self, cam_name)
            cam.attach(self.robot._robot_entity.get_link(link_name), offset)
            cam.move_to_attach()

    def _init_reward_functions(self) -> None:
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.ctrl_dt
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
        random_y = (th.rand(num_reset, device=self.device) - 0.5) * 0.5  # -0.25 – 0.25
        random_z = th.ones(num_reset, device=self.device) * 0.025  # 0.15 – 0.15
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
                / self.cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

    def reset(self) -> tuple[TensorDict, dict]:
        self.reset_buf[:] = True
        self.reset_idx(th.arange(self.num_envs, device=gs.device))
        return self.get_observations(), self.extras

    def step(self, actions: th.Tensor) -> tuple[TensorDict, th.Tensor, th.Tensor, dict]:
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
        reward = th.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
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
        match self.camera_setup:
            case "default":
                cams = [self.scene_cam, self.tool_left_cam, self.tool_right_cam]
            case "scene_dual":
                cams = [self.scene_left_cam, self.scene_right_cam]
            case _:
                raise ValueError(f"Unknown camera setup {self.camera_setup}")

        rgb_list = []
        for cam in cams:
            rgb, _, _, _ = cam.render(
                rgb=True, depth=False, segmentation=False, normal=False
            )
            rgb = rgb.permute(0, 3, 1, 2)[:, :3]
            if normalize:
                rgb = th.clamp(rgb, 0.0, 255.0) / 255.0
            rgb_list.append(rgb)

        rgb_multi = th.cat(rgb_list, dim=1)
        return rgb_multi

    def _reward_keypoints(self) -> th.Tensor:
        ee_pos = self.robot.ee_pose[:, :3]
        ee_quat = self.robot.ee_pose[:, 3:7]
        keypoints_offset = self.keypoints_offset
        # there is a offset between the finger tip and the finger base frame
        # finger_tip_z_offset = th.tensor(
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
        reset_pose = th.tensor(
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
