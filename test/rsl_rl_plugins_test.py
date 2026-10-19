"""
These classes are loaded by name from the YAML configs, so nothing else imports them and a typo
would only surface mid-training.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th
import yaml
from rsl_rl.utils import LogWriter, resolve_callable
from tensordict import TensorDict

from aegis_gym.aux.logging.clearml_writer import ClearMLLogWriter
from aegis_gym.runners.models import AegisMLPModel
from aegis_gym.runners.on_policy_runner import CLEARML_WRITER

CONFIG_DIR = Path(__file__).parent.parent / "aegis_gym/config"


def _rl_section(path):
    data = yaml.safe_load((CONFIG_DIR / path).read_text())
    return data.get("rl", data)


RL_CONFIGS = ["defaults/rl.yaml", "presets/push_t.yaml", "presets/reacher.yaml"]


@pytest.mark.parametrize("config", RL_CONFIGS)
@pytest.mark.parametrize("model", ["actor", "critic"])
def test_model_class_name_in_config_resolves(config, model):
    resolved = resolve_callable(_rl_section(config)[model]["class_name"])
    assert issubclass(resolved, AegisMLPModel)


@pytest.mark.parametrize("config", RL_CONFIGS)
def test_obs_groups_use_the_actor_set(config):
    obs_groups = _rl_section(config)["obs_groups"]
    assert "actor" in obs_groups
    assert "policy" not in obs_groups


def test_clearml_writer_class_name_resolves():
    assert resolve_callable(CLEARML_WRITER) is ClearMLLogWriter


def test_clearml_writer_implements_the_log_writer_interface():
    assert issubclass(ClearMLLogWriter, LogWriter)
    assert not getattr(ClearMLLogWriter, "__abstractmethods__", set())


def test_clearml_writer_drops_the_time_scalars():
    """These series would report elapsed seconds where ClearML expects an iteration."""
    source = inspect.getsource(ClearMLLogWriter.add_scalar)
    assert '"/time"' in source


def _make_model(**kwargs) -> AegisMLPModel:
    obs = TensorDict({"policy": th.zeros(2, 4)}, batch_size=[2])
    return AegisMLPModel(
        obs,
        {"actor": ["policy"], "critic": ["policy"]},
        "actor",
        3,
        hidden_dims=[8, 8],
        **kwargs,
    )


@pytest.mark.parametrize("detach", [True, False])
def test_detach_actor_grad_cuts_the_gradient_to_the_observations(detach):
    model = _make_model(detach_actor_grad=detach)
    obs = TensorDict({"policy": th.zeros(2, 4, requires_grad=True)}, batch_size=[2])

    model(obs).sum().backward()

    reached_obs = obs["policy"].grad is not None
    assert reached_obs is not detach


def test_reset_last_layer_weights_only_touches_the_last_layer():
    model = _make_model()
    linears = [m for m in model.mlp if isinstance(m, th.nn.Linear)]
    first_before = linears[0].weight.clone()
    last_before = linears[-1].weight.clone()
    linears[-1].bias.data.fill_(0.5)

    model.reset_last_layer_weights()

    assert th.equal(linears[0].weight, first_before), "an earlier layer was reset"
    assert not th.equal(linears[-1].weight, last_before), "the last layer was not reset"
    assert not linears[-1].bias.any(), "the last layer's bias was not zeroed"


def test_clearml_writer_stop_keeps_the_task_and_the_writer_open():
    """Training calls this once per chunk, so closing here would silence every later chunk."""
    source = inspect.getsource(ClearMLLogWriter.stop)
    assert "close()" not in source
    assert "flush()" in source


def _writer_with_fake_task(log_cfg_as_hyperparams: bool):
    """Built without the constructor, which would open a real ClearML task."""
    writer = object.__new__(ClearMLLogWriter)
    writer._log_cfg_as_hyperparams = log_cfg_as_hyperparams
    connected: dict[str, object] = {}
    writer.task = SimpleNamespace(
        connect=lambda cfg, name: connected.__setitem__(name, cfg)
    )
    return writer, connected


TRAIN_CFG = {
    "num_steps_per_env": 24,
    "actor": {"hidden_dims": [8]},
    "critic": {"hidden_dims": [8]},
    "algorithm": {"gamma": 0.99},
}


def test_store_config_connects_every_section():
    writer, connected = _writer_with_fake_task(log_cfg_as_hyperparams=True)

    writer.store_config({"num_envs": 4}, TRAIN_CFG)

    assert set(connected) == {
        "runner_cfg",
        "actor_cfg",
        "critic_cfg",
        "alg_cfg",
        "env_cfg",
    }
    assert connected["actor_cfg"] == {"hidden_dims": [8]}
    assert connected["alg_cfg"] == {"gamma": 0.99}
    assert set(connected["runner_cfg"]) == {"num_steps_per_env"}


def test_store_config_honours_the_disable_flag():
    writer, connected = _writer_with_fake_task(log_cfg_as_hyperparams=False)

    writer.store_config({"num_envs": 4}, TRAIN_CFG)

    assert connected == {}
