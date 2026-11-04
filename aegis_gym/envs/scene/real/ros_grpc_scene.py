import math
import time
from typing import Any

import torch as th

from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import Control, ExpConfig, RobotCfg
from aegis_gym.envs.manipulator import BaseManipulator, RosGrpcManipulator
from aegis_gym.envs.objects import (
    BaseObject,
    ObjectProperties,
    ObjectType,
    RosGrpcObjectsFactory,
)

from ..base_scene import BaseScene

OVERRUN_TOLERANCE_S = 0.002
# Longer than this between the observation and the next action is a pause (reset, MoveIt), not an overrun.
PAUSE_S = 0.5


class RosGrcpScene(BaseScene):
    def __init__(
        self,
        cfg: ExpConfig,
        device: th.device,
    ):
        logger = get_logger("RosGrpcScene")
        self._logger = logger
        if cfg.env_cfg.num_envs > 1:
            raise ValueError("num_envs > 1 not supported for single robot station")

        cfg_env = cfg.env_cfg
        cfg_dr = cfg.dr_cfg
        super().__init__(device=device, enable_plotjuggler=cfg.args.enable_plotjuggler)
        self.CONTROL_TYPE = Control.ROS
        self._randomization_fns = {
            # TODO(issue#139) implement scene lighting randomization
            # RandomizationType.SCENE_LIGHTING: self._rand_scene_lighting,
        }

        self._cfg_env = cfg_env
        self._cfg_dr = cfg_dr
        self.disable_cameras = cfg.args.disable_cameras
        self._extract_config()

        logger.info(
            f"f_c: {1 / self.ctrl_dt} Hz | f_pi: {1 / self.policy_dt} Hz | Action: {self.sim_substeps} steps | Max speed: {self.max_linear_speed} m/s ; {self.max_angular_speed} rad/s"
        )

        self._global_entity_cnt = 0
        self._entity_registry: dict[int, BaseObject] = {}

    def _shutdown(self) -> None:
        self.manipulator.shutdown()

    def _extract_config(self) -> None:
        self.last_step_ts: float = time.perf_counter()
        self._obs_ts: float | None = None

        self.num_obs = self._cfg_env.num_obs
        self.num_privileged_obs = None
        self.num_actions = self._cfg_env.num_actions
        self.image_width = self._cfg_env.image_resolution[0]
        self.image_height = self._cfg_env.image_resolution[1]
        self.rgb_image_shape = (3, self.image_height, self.image_width)
        self.cameras_setup = self._cfg_env.cameras_setup
        self.table_size = self._cfg_env.table_size
        self.workbench_size = self._cfg_env.workbench_size

        self.ctrl_dt = self._cfg_env.ctrl_dt
        self.policy_dt = self._cfg_env.policy_dt
        self.obs_lead_s = min(max(self._cfg_env.obs_lead_s, 0.0), self.policy_dt)
        self._overruns = 0
        self.sim_substeps = math.ceil(self._cfg_env.policy_dt / self._cfg_env.ctrl_dt)
        self.max_episode_length = math.ceil(
            self._cfg_env.episode_length_s / self.policy_dt
        )

        self.max_linear_speed = self._cfg_env.action_max_linear_speed
        self.max_angular_speed = self._cfg_env.action_max_angular_speed

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
            disable_cameras=self.disable_cameras,
            device=self.device,
        )

    def _build(self) -> None:
        pass

    def _get_manipulator(self) -> BaseManipulator:
        return self.manipulator

    def _update_state(self) -> None:
        self.manipulator.read_state()

    @staticmethod
    def _sleep_until(deadline: float) -> None:
        while (remaining := deadline - time.perf_counter()) > 0:
            time.sleep(min(remaining, 0.0005))

    def pre_step(self) -> None:
        """Paces the actions on a fixed `policy_dt` grid.

        A late action means the previous trajectory segment has already ended and the arm
        stood still in between; it is counted. After a late action or a pause the grid
        restarts from now.
        """
        now = time.perf_counter()
        deadline = self.last_step_ts + self.policy_dt
        late = now - deadline
        paused = self._obs_ts is None or now - self._obs_ts > PAUSE_S
        if late > OVERRUN_TOLERANCE_S and not paused:
            self._overruns += 1
            if self._overruns == 1 or self._overruns % 100 == 0:
                self._logger.warning(
                    f"Action {late * 1e3:.1f} ms late ({self._overruns} so far): "
                    f"read + inference exceed obs_lead_s = {self.obs_lead_s * 1e3:.0f} ms"
                )
        self._sleep_until(deadline)
        self.last_step_ts = (
            deadline if late <= OVERRUN_TOLERANCE_S else time.perf_counter()
        )

    def step(self) -> None:
        """Reads the state `obs_lead_s` before the next action is due rather than right after
        this one, so the policy acts on a fresh observation; the arm executes the segment
        meanwhile. This also matches the simulation, which observes after a full `policy_dt`."""
        self._sleep_until(self.last_step_ts + self.policy_dt - self.obs_lead_s)
        self.update_state()
        self._obs_ts = time.perf_counter()

    def _rand_scene_lighting(self, envs_idx: th.Tensor) -> None:
        raise NotImplementedError("TODO(issue#139) implement scene lighting")
