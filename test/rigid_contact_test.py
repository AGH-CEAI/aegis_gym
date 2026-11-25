from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from aegis_gym.config import ConfigManager as cm
from aegis_gym.config.types.env import RIGID_CONTACT_KEYS, validate_rigid_contact

# set by `GenesisScene._setup_create_scene` itself, overriding them would break the scene
_SCENE_MANAGED_KEYS = (
    "dt",
    "constraint_solver",
    "enable_collision",
    "enable_joint_limit",
    "batch_dofs_info",
    "batch_links_info",
)
_SAMPLE = {
    "box_box_detection": True,
    "constraint_timeconst": 0.02,
    "noslip_iterations": 5,
    "iterations": 100,
}


@pytest.fixture(autouse=True)
def reset_config():
    cm._global_cfg = None


def _setup_with_file(tmp_path, env: dict) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"env": env}))
    cm.setup_config(argv=["", "-c", str(path)])


def test_default_keeps_genesis_defaults():
    cm.setup_config(argv=[""])
    assert cm.get_config().env_cfg.rigid_contact is None


def test_config_file_overrides_are_applied(tmp_path):
    _setup_with_file(tmp_path, {"rigid_contact": _SAMPLE})
    assert cm.get_config().env_cfg.rigid_contact == _SAMPLE


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError, match="box_box_detecton"):
        _setup_with_file(tmp_path, {"rigid_contact": {"box_box_detecton": True}})


@pytest.mark.parametrize("key", _SCENE_MANAGED_KEYS)
def test_scene_managed_keys_are_rejected(key):
    with pytest.raises(ValueError, match=key):
        validate_rigid_contact({key: 0.001})


@pytest.mark.parametrize(
    "overrides",
    [
        {"box_box_detection": 1},  # int is not a bool
        {"iterations": 10.5},
        {"iterations": True},  # bool is not an int
        {"constraint_timeconst": "0.01"},
    ],
)
def test_wrong_types_raise(overrides):
    with pytest.raises(ValueError, match=next(iter(overrides))):
        validate_rigid_contact(overrides)


def test_ints_are_accepted_as_floats():
    validate_rigid_contact({"constraint_timeconst": 1})


def test_keys_exist_in_genesis_with_the_same_types():
    gs = pytest.importorskip("genesis")
    fields = gs.options.RigidOptions.model_fields
    for key, typ in RIGID_CONTACT_KEYS.items():
        assert key in fields, f"{key} is not a Genesis RigidOptions field"
        assert typ.__name__ in str(fields[key].annotation), (
            f"{key}: {fields[key].annotation} vs {typ}"
        )
    assert not set(_SCENE_MANAGED_KEYS) & set(RIGID_CONTACT_KEYS)


def test_scene_passes_the_overrides_to_genesis(monkeypatch):
    pytest.importorskip("genesis")
    from aegis_gym.envs.scene.sim import genesis_scene

    captured = {}
    monkeypatch.setattr(
        genesis_scene.gs, "Scene", lambda **kw: captured.update(kw) or MagicMock()
    )
    scene = object.__new__(genesis_scene.GenesisScene)
    scene.use_cameras = False
    scene.policy_dt, scene.ctrl_dt, scene.sim_substeps, scene.num_envs = (
        0.04,
        0.004,
        10,
        1,
    )
    scene._preview_enabled, scene._preview_envs_idx = False, [0]
    env_cfg = SimpleNamespace(use_rasterizer=True, rigid_contact=_SAMPLE)

    scene._setup_create_scene(env_cfg, show_viewer=False)

    opts = captured["rigid_options"]
    for key, value in _SAMPLE.items():
        assert getattr(opts, key) == value
    assert opts.dt == 0.004  # the scene's own settings stay
    assert opts.batch_dofs_info is True


def test_scene_without_overrides_matches_genesis_defaults(monkeypatch):
    gs = pytest.importorskip("genesis")
    from aegis_gym.envs.scene.sim import genesis_scene

    captured = {}
    monkeypatch.setattr(
        genesis_scene.gs, "Scene", lambda **kw: captured.update(kw) or MagicMock()
    )
    scene = object.__new__(genesis_scene.GenesisScene)
    scene.use_cameras = False
    scene.policy_dt, scene.ctrl_dt, scene.sim_substeps, scene.num_envs = (
        0.04,
        0.004,
        10,
        1,
    )
    scene._preview_enabled, scene._preview_envs_idx = False, [0]

    scene._setup_create_scene(
        SimpleNamespace(use_rasterizer=True, rigid_contact=None), show_viewer=False
    )

    opts, defaults = captured["rigid_options"], gs.options.RigidOptions.model_fields
    for key in RIGID_CONTACT_KEYS:
        assert getattr(opts, key) == defaults[key].default
