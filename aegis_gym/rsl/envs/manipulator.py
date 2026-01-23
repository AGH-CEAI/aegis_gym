from typing import Literal

import torch as th
import genesis as gs
from genesis.utils.geom import (
    transform_quat_by_quat,
    xyz_to_quat,
)

from .utils import generate_aegis_urdf


class Manipulator:
    def __init__(
        self,
        num_envs: int,
        scene: gs.Scene,
        args: dict,
        show_cell: bool,
        device: str = "cpu",
    ):
        # == set members ==
        self._device = device
        self._scene = scene
        self._num_envs = num_envs
        self._args = args

        # == Genesis configurations ==
        material: gs.materials.Rigid = gs.materials.Rigid()
        morph: gs.morphs.URDF = gs.morphs.URDF(
            file=generate_aegis_urdf(show_cell),
            fixed=True,
            pos=(0.0, 0.0, 0.0),
            quat=(1.0, 0.0, 0.0, 0.0),
            links_to_keep=[
                "ur_base",
                "robotiq_hande_end",
                "cam_tool_right",
                "cam_tool_left",
                "cam_scene_rgb_camera_frame",
            ],
        )
        self._robot_entity: gs.Entity = scene.add_entity(material=material, morph=morph)

        self._gripper_open_dof = 0.025
        self._gripper_close_dof = 0.0

        self._ik_method: Literal["rel_pose", "dls"] = args["ik_method"]

        self._init()

    def set_pd_gains(self):
        # set control gains
        self._robot_entity.set_dofs_kp(
            th.tensor([4500, 4500, 3500, 3500, 2000, 2000, 100, 100]),
        )
        self._robot_entity.set_dofs_kv(
            th.tensor([450, 450, 350, 350, 200, 200, 10, 10]),
        )
        self._robot_entity.set_dofs_force_range(
            th.tensor([-87, -87, -87, -87, -12, -12, -100, -100]),
            th.tensor([87, 87, 87, 87, 12, 12, 100, 100]),
        )

    def _init(self):
        self._arm_dof_dim = self._robot_entity.n_dofs - 2  # total number of arm joints
        self._gripper_dim = 2  # number of gripper joints

        self._arm_dof_idx = th.arange(self._arm_dof_dim, device=self._device)
        self._fingers_dof = th.arange(
            self._arm_dof_dim,
            self._arm_dof_dim + self._gripper_dim,
            device=self._device,
        )
        self._left_finger_dof = self._fingers_dof[0]
        self._right_finger_dof = self._fingers_dof[1]
        self._ee_link = self._robot_entity.get_link(self._args["ee_link_name"])
        # self._left_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][0])
        # self._right_finger_link = self._robot_entity.get_link(self._args["gripper_link_names"][1])
        self._default_joint_angles = self._args["default_arm_dof"]
        if self._args["default_gripper_dof"] is not None:
            self._default_joint_angles += self._args["default_gripper_dof"]

    def reset(self, envs_idx: th.IntTensor):
        if len(envs_idx) == 0:
            return
        self.reset_home(envs_idx)

    def reset_home(self, envs_idx: th.IntTensor | None = None):
        if envs_idx is None:
            envs_idx = th.arange(self._num_envs, device=self._device)
        default_joint_angles = th.tensor(
            self._default_joint_angles, dtype=th.float32, device=self._device
        ).repeat(len(envs_idx), 1)
        self._robot_entity.set_qpos(default_joint_angles, envs_idx=envs_idx)

    def apply_action(self, action: th.Tensor, open_gripper: bool) -> None:
        """
        Apply the action to the robot.
        """
        q_pos = self._robot_entity.get_qpos()
        if self._ik_method == "gs_ik":
            q_pos = self._gs_ik(action)
        elif self._ik_method == "dls_ik":
            q_pos = self._dls_ik(action)
        else:
            raise ValueError(f"Invalid control mode: {self._ik_method}")
        # set gripper to open
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    def _gs_ik(self, action: th.Tensor) -> th.Tensor:
        """
        Genesis inverse kinematics
        """
        delta_position = action[:, :3]
        delta_orientation = action[:, 3:6]

        # compute target pose
        target_position = delta_position + self._ee_link.get_pos()
        quat_rel = xyz_to_quat(delta_orientation, rpy=True, degrees=False)
        target_orientation = transform_quat_by_quat(quat_rel, self._ee_link.get_quat())
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=target_position,
            quat=target_orientation,
            dofs_idx_local=self._arm_dof_idx,
        )
        return q_pos

    def _dls_ik(self, action: th.Tensor) -> th.Tensor:
        """
        Damped least squares inverse kinematics
        """
        delta_pose = action[:, :6]
        lambda_val = 0.01
        jacobian = self._robot_entity.get_jacobian(link=self._ee_link)
        jacobian_T = jacobian.transpose(1, 2)
        lambda_matrix = (lambda_val**2) * th.eye(
            n=jacobian.shape[1], device=self._device
        )
        delta_joint_pos = (
            jacobian_T
            @ th.inverse(jacobian @ jacobian_T + lambda_matrix)
            @ delta_pose.unsqueeze(-1)
        ).squeeze(-1)
        return self._robot_entity.get_qpos() + delta_joint_pos

    def go_to_goal(self, goal_pose: th.Tensor, open_gripper: bool = True):
        q_pos = self._robot_entity.inverse_kinematics(
            link=self._ee_link,
            pos=goal_pose[:, :3],
            quat=goal_pose[:, 3:7],
            dofs_idx_local=self._arm_dof_idx,
        )
        if open_gripper:
            q_pos[:, self._fingers_dof] = self._gripper_open_dof
        else:
            q_pos[:, self._fingers_dof] = self._gripper_close_dof
        self._robot_entity.control_dofs_position(position=q_pos)

    @property
    def base_pos(self):
        return self._robot_entity.get_pos()

    @property
    def ee_pose(self) -> th.Tensor:
        """
        The end-effector pose (the hand pose)
        """
        pos, quat = self._ee_link.get_pos(), self._ee_link.get_quat()
        return th.cat([pos, quat], dim=-1)

    # @property
    # def left_finger_pose(self) -> th.Tensor:
    #     pos, quat = self._left_finger_link.get_pos(), self._left_finger_link.get_quat()
    #     return th.cat([pos, quat], dim=-1)

    # @property
    # def right_finger_pose(self) -> th.Tensor:
    #     pos, quat = (
    #         self._right_finger_link.get_pos(),
    #         self._right_finger_link.get_quat(),
    #     )
    #     return th.cat([pos, quat], dim=-1)

    # @property
    # def center_finger_pose(self) -> th.Tensor:
    #     """
    #     The center finger pose is the average of the left and right finger poses.
    #     """
    #     left_finger_pose = self.left_finger_pose
    #     right_finger_pose = self.right_finger_pose
    #     center_finger_pos = (left_finger_pose[:, :3] + right_finger_pose[:, :3]) / 2
    #     center_finger_quat = left_finger_pose[:, 3:7]
    #     return th.cat([center_finger_pos, center_finger_quat], dim=-1)
