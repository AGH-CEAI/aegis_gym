from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from aegis_gym.aux.action_preview import ActionPreviewRenderer, ActionSpec
from aegis_gym.config.types import DebugCfg
from aegis_gym.envs import PushTEnv, ReacherEnv
from aegis_gym.envs.base_env import BaseEnv, StepReturn
from aegis_gym.envs.wrappers import action_preview_wrapper
from aegis_gym.envs.wrappers.action_preview_wrapper import ActionPreviewEnvWrapper

PLANAR_SPEC = ActionSpec(
    names=("vx", "vy"), units=("m/s", "m/s"), scales=(0.1, 0.1), joystick=(0, 1)
)


def test_action_spec_lengths_must_match():
    with pytest.raises(ValueError, match="2 names, 1 units"):
        ActionSpec(names=("vx", "vy"), units=("m/s",), scales=(0.1, 0.1))


def test_action_spec_joystick_must_be_of_its_actions():
    with pytest.raises(ValueError, match="joystick"):
        ActionSpec(names=("vx",), units=("m/s",), scales=(0.1,), joystick=(0, 1))


def test_action_spec_default_and_dict_round_trip():
    assert ActionSpec.default(3).names == ("a0", "a1", "a2")
    assert ActionSpec.from_dict(PLANAR_SPEC.as_dict()) == PLANAR_SPEC


@pytest.mark.parametrize(
    ("spec", "width"),
    [
        (ActionSpec.default(6), ActionPreviewRenderer.STRIPS_WIDTH),
        (
            PLANAR_SPEC,
            ActionPreviewRenderer.STRIPS_WIDTH + ActionPreviewRenderer.JOYSTICK_WIDTH,
        ),
    ],
)
def test_renderer_draws_a_strip_per_action(spec: ActionSpec, width: int):
    renderer = ActionPreviewRenderer(spec, max_steps=10, policy_dt=0.04, title="test")
    num_actions = len(spec.names)
    for step in range(5):
        renderer.add(step=step, raw=[1.5] * num_actions, applied=[1.0] * num_actions)

    img = renderer.render(paused=True)

    strips = ActionPreviewRenderer.STRIP_HEIGHT * num_actions
    assert img.shape[1] == width
    assert img.shape[0] >= ActionPreviewRenderer.HEADER_HEIGHT + strips
    assert img.dtype == np.uint8


def test_renderer_starts_the_next_episode_after_a_done_or_a_reset():
    renderer = ActionPreviewRenderer(
        PLANAR_SPEC, max_steps=10, policy_dt=0.04, title="t"
    )
    renderer.add(step=0, raw=[0.1, 0.2])
    renderer.add(step=1, raw=[0.3, 0.4], done=True, reason="success")
    # done: the step 2 is of the next episode, as is a step not following the last one
    renderer.add(step=2, raw=[0.0, 0.0])
    renderer.add(step=0, raw=[0.5, 0.5])

    assert renderer._episode == 3
    assert renderer._last_end_reason == "success"
    # the previous episode is the 2nd one, of its step 2 only
    assert np.isnan(renderer._previous[:2]).all()
    assert renderer._previous[2].tolist() == [0.0, 0.0]
    assert renderer._raw[0].tolist() == [0.5, 0.5]


def test_renderer_keeps_the_steps_past_the_episode_length_and_the_gaps():
    renderer = ActionPreviewRenderer(
        PLANAR_SPEC, max_steps=4, policy_dt=0.04, title="t"
    )
    for step in (0, 1, 3, 8):  # the steps 2 and 4-7 were dropped
        renderer.add(step=step, raw=[0.1, -0.1], applied=[0.1, -0.1])

    renderer.render()

    assert np.isnan(renderer._raw[2]).all()
    assert renderer._raw[8].tolist() == [0.1, -0.1]
    assert renderer._episode == 1


class _Client:
    def __init__(self, **kwargs) -> None:
        self.kwargs, self.sent, self.is_open = kwargs, [], True

    def send(self, **row) -> None:
        self.sent.append(row)

    def close(self) -> None:
        self.is_open = False


class _Env:
    """Two planar envs, the env 1 ends its episode with a success on the 2nd step."""

    def __init__(self) -> None:
        self.num_envs, self.num_actions, self.max_episode_length = 2, 2, 10
        self.episode_length_buf = th.zeros(2)
        self.applied_actions = None
        self._scene = None  # checked by `BaseEnv.__del__()` of the wrapper

    def get_action_spec(self) -> ActionSpec:
        return PLANAR_SPEC

    def get_policy_dt(self) -> float:
        return 0.04

    def step(self, actions: th.Tensor) -> StepReturn:
        self.episode_length_buf += 1
        self.applied_actions = th.clamp(actions, -1.0, 1.0)
        done = self.episode_length_buf >= 2
        self.episode_length_buf[done] = 0
        extras = {"success": done.clone(), "time_outs": th.zeros(2)}
        return StepReturn(None, th.zeros(2), done, extras)


@pytest.fixture
def wrapper(monkeypatch) -> ActionPreviewEnvWrapper:
    monkeypatch.setattr(action_preview_wrapper, "ActionPreviewClient", _Client)
    cfg = DebugCfg(enabled=True, enable_action_preview=True, action_preview_env_idx=1)
    return ActionPreviewEnvWrapper(env=_Env(), cfg_debug=cfg)


def test_wrapper_sends_the_raw_and_applied_actions_of_its_env(wrapper):
    wrapper.step(th.tensor([[0.0, 0.0], [2.0, -0.5]]))
    wrapper.step(th.tensor([[0.0, 0.0], [0.3, 0.1]]))
    wrapper.step(th.tensor([[0.0, 0.0], [0.2, 0.2]]))

    sent = wrapper._client.sent
    assert [row["step"] for row in sent] == [0, 1, 0]
    assert sent[0]["raw"] == pytest.approx([2.0, -0.5])
    assert sent[0]["applied"] == pytest.approx([1.0, -0.5])
    assert [row["done"] for row in sent] == [False, True, False]
    assert sent[1]["reason"] == "success"
    assert wrapper._client.kwargs["max_steps"] == 10


def test_wrapper_stops_sending_once_the_window_is_closed(wrapper):
    wrapper.close()

    wrapper.step(th.zeros(2, 2))

    assert wrapper._client.sent == []


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (DebugCfg(enabled=False, enable_action_preview=True), "--debug-enable"),
        (DebugCfg(enabled=True, action_preview_env_idx=2), "out of the 2 envs"),
    ],
)
def test_wrapper_rejects_a_wrong_config(monkeypatch, cfg: DebugCfg, match: str):
    monkeypatch.setattr(action_preview_wrapper, "ActionPreviewClient", _Client)
    with pytest.raises(ValueError, match=match):
        ActionPreviewEnvWrapper(env=_Env(), cfg_debug=cfg)


def test_wrapper_rejects_an_action_spec_of_other_actions(monkeypatch):
    monkeypatch.setattr(action_preview_wrapper, "ActionPreviewClient", _Client)
    env = _Env()
    env.get_action_spec = lambda: ActionSpec.default(3)

    with pytest.raises(ValueError, match="describes 3 actions, the env has 2"):
        ActionPreviewEnvWrapper(env=env, cfg_debug=DebugCfg(enabled=True))


@pytest.mark.parametrize(
    ("env_cls", "num_actions", "joystick"),
    [(PushTEnv, 2, ("vx", "vy")), (ReacherEnv, 6, ("vx", "vy"))],
)
def test_env_action_spec_describes_its_actions(
    env_cls: type, num_actions: int, joystick: tuple[str, str]
):
    env = object.__new__(env_cls)
    env.max_linear_speed, env.max_angular_speed = 0.1, 0.2

    spec = env.get_action_spec()

    assert len(spec.names) == num_actions
    assert tuple(spec.names[i] for i in spec.joystick) == joystick
    assert spec.scales[0] == 0.1
    assert all(scale > 0 for scale in spec.scales)


def test_base_env_action_spec_defaults_to_unnamed_actions():
    env = SimpleNamespace(num_actions=4)

    spec = BaseEnv.get_action_spec(env)

    assert spec == ActionSpec.default(4)
