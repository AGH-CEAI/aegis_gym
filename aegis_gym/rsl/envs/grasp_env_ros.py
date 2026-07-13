from .manipulator import RosGrpcManipulator
from .base_env import BaseEnv, Modality
from .objects import BaseBox, ObjectsFactory

from config import ExpConfig
from config.types import Control, CamerasSetup

from .scene import BaseScene


class GraspEnvROS(BaseEnv):
    DEFAULT_MODALITIES = frozenset({Modality.TCP_POSE, Modality.OBJECT_POSE})

    def __init__(self, cfg: ExpConfig, scene: BaseScene = None) -> None:
        super().__init__(scene=scene, cfg=cfg)
        self._extract_config()
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
