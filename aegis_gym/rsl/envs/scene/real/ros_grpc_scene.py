import time
from typing import Optional, Any

# TODO try to remove the np dependnecy
import torch as th

from ..base_scene import BaseScene, RandomizationType
from envs.manipulator import BaseManipulator, RosGrpcManipulator
from envs.objects import ObjectType, BaseObject, ObjectProperties
from config.types import CameraName, Control, ExpConfig, RobotCfg


class RosGrcpScene(BaseScene):
    def __init__(
        self,
        cfg: ExpConfig,
        device: th.device,
    ):
        cfg_env = cfg.env_cfg
        cfg_dr = cfg.dr_cfg
        super().__init__(device=device)
        self.CONTROL_TYPE = Control.ROS
        self._randomization_fns = {
            RandomizationType.SCENE_LIGHTING: self._rand_scene_lighting,
        }

        self._cfg_env = cfg_env
        self._cfg_dr = cfg_dr
        self.disable_vision = cfg.args.disable_vision
        self._extract_config()

        print(
            f"[ROSgRPCScene] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self._max_linear_speed} m/s ; {self._max_angular_speed} rad/s"
        )

        # TODO change Any to CameraType?
        self._cameras: dict[CameraName, Any] = {}
        self._setup_scene(cfg=cfg)

        self._global_entity_cnt = 0
        self._entity_registry: dict[int, BaseObject] = {}

    def _extract_config(self) -> None:
        self.last_step_ts: Optional[float] = None
        raise NotImplementedError

    def _setup_scene(self, cfg: ExpConfig) -> None:
        raise NotImplementedError

    def get_policy_dt(self) -> float:
        raise NotImplementedError

    def add_entity(self, entity: ObjectType, properties: ObjectProperties) -> Any:
        raise NotImplementedError

    def add_manipulator(self, cfg: RobotCfg) -> None:
        self.manipulator = RosGrpcManipulator(
            num_envs=self.num_envs,
            scene=self,
            robot_cfg=cfg.robot_cfg,
            policy_dt=cfg.env_cfg.policy_dt,
            disable_vision=self.disable_vision,
            device=self.device,
        )

    def _build(self) -> None:
        pass

    def _get_manipulator(self) -> BaseManipulator:
        return self.manipulator

    def update_state(self) -> None:
        raise NotImplementedError

    def step(self) -> None:
        # TODO WARNING: THIS SHOULD BE EXECUTED AFTER APPLYING COMMAND TO THE ROBOT. IN GENESIS THIS LOGIC SI NVERTED!
        # TODO make this in API with  flag to enable/disable
        if not self.last_step_ts:
            self.last_step_ts = time.perf_counter()

        while time.perf_counter() - self.last_step_ts < self.policy_dt:
            time.sleep(0.0001)
        self.last_step_ts = time.perf_counter()

    def _rand_scene_lighting(self, envs_idx: th.Tensor) -> None:
        pass
