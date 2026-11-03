import torch as th

from aegis_gym.aux.action_preview.viewer import ActionPreviewClient
from aegis_gym.aux.logging import get_logger
from aegis_gym.config.types import DebugCfg

from ..base_env import BaseEnv, StepReturn
from .base_wrapper import BaseEnvWrapper


class ActionPreviewEnvWrapper(BaseEnvWrapper):
    """
    The debug wrapper showing the actions of the env `DebugCfg.action_preview_env_idx` live, in a
    window of its own process: the raw (policy) and the applied actions over the whole episode.
    """

    def __init__(self, env: BaseEnv, cfg_debug: DebugCfg):
        super().__init__(env=env)
        if not cfg_debug.enabled:
            raise ValueError(
                "Activate the debugging mode to use the ActionPreviewEnvWrapper (`--debug-enable`)."
            )
        self._env_idx = cfg_debug.action_preview_env_idx
        if not 0 <= self._env_idx < env.num_envs:
            raise ValueError(
                f"The action preview env {self._env_idx} is out of the {env.num_envs} envs."
            )
        spec = env.get_action_spec()
        if len(spec.names) != env.num_actions:
            raise ValueError(
                f"The action spec describes {len(spec.names)} actions, the env has "
                f"{env.num_actions}."
            )
        get_logger("ActionPreviewEnvWrapper").info(
            f"Showing the actions of the env {self._env_idx}: {', '.join(spec.names)}"
        )
        self._client = ActionPreviewClient(
            spec=spec,
            max_steps=int(env.max_episode_length),
            policy_dt=env.get_policy_dt(),
            title=f"[DEBUG] Action preview (env {self._env_idx})",
            save_dir=cfg_debug.record_dir,
        )

    def step(self, actions: th.Tensor) -> StepReturn:
        if not self._client.is_open:
            return self._env.step(actions)
        idx = self._env_idx
        # read before the step, so the episodes are told apart after any kind of reset
        step = int(self._env.episode_length_buf[idx])
        raw = actions[idx].detach().float().cpu().tolist()
        res = self._env.step(actions)

        applied = self._env.applied_actions
        if applied is not None:
            applied = applied[idx].float().cpu().tolist()
        done = bool(res.dones[idx])
        self._client.send(
            step=step,
            raw=raw,
            applied=applied,
            done=done,
            reason=self._end_reason(res.extras, idx) if done else None,
        )
        return res

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _end_reason(extras: dict, idx: int) -> str:
        for key, reason in (("success", "success"), ("time_outs", "timeout")):
            value = extras.get(key)
            if isinstance(value, th.Tensor) and bool(value[idx]):
                return reason
        return "terminated"
