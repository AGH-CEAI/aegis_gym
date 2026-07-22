import math
import time
from typing import Any

import torch as th

from ..base_scene import BaseScene
from envs.manipulator import BaseManipulator, RosGrpcManipulator
from envs.objects import ObjectType, BaseObject, ObjectProperties, RosGrpcObjectsFactory
from config.types import Control, ExpConfig, RobotCfg


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
            # TODO(issue#139) implement scene lighting randomization
            # RandomizationType.SCENE_LIGHTING: self._rand_scene_lighting,
        }

        self._cfg_env = cfg_env
        self._cfg_dr = cfg_dr
        self.disable_vision = cfg.args.disable_vision
        self._extract_config()

        print(
            f"[RosGrpcScene] f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self._global_entity_cnt = 0
        self._entity_registry: dict[int, BaseObject] = {}

    def shutdown(self) -> None:
        self.manipulator.shutdown()

    def _extract_config(self) -> None:
        self.last_step_ts: float = time.perf_counter()

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

    def get_policy_dt(self) -> float:
        return self.policy_dt

    def add_entity(self, entity: ObjectType, properties: ObjectProperties) -> Any:
        f = RosGrpcObjectsFactory(device=self.device)
        obj = f.create_object(
            obj_type=entity,
            obj_properties=properties,
        )
        self._entity_registry[self._global_entity_cnt] = obj
        self._global_entity_cnt += 1
        return obj

    def add_manipulator(self, cfg: RobotCfg) -> None:
        self.manipulator = RosGrpcManipulator(
            num_envs=1,
            robot_cfg=cfg,
            policy_dt=self.policy_dt,
            disable_vision=self.disable_vision,
            device=self.device,
        )

    def _build(self) -> None:
        pass

    def _get_manipulator(self) -> BaseManipulator:
        return self.manipulator

    def update_state(self) -> None:
        self.manipulator.read_state()

    def pre_step(self) -> None:
        while time.perf_counter() - self.last_step_ts < self.policy_dt:
            time.sleep(0.0001)
        self.last_step_ts = time.perf_counter()

    def step(self) -> None:
        self.update_state()

    def _rand_scene_lighting(self, envs_idx: th.Tensor) -> None:
        raise NotImplementedError("TODO(issue#139) implement scene lighting")
