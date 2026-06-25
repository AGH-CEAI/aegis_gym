from dataclasses import fields

import pytest

from config import LaunchArgs, parse_arguments
from config import ConfigManager as cm
from config.types import Stage, Control


def dict_diff(actual: dict, expected: dict, path: str = ""):
    diffs = []

    all_keys = set(actual) | set(expected)

    for key in all_keys:
        current_path = f"{path}.{key}" if path else str(key)

        if key not in actual:
            diffs.append(f"Missing in actual: {current_path}")
            continue

        if key not in expected:
            diffs.append(f"Unexpected key: {current_path}")
            continue

        a = actual[key]
        e = expected[key]

        if isinstance(a, dict) and isinstance(e, dict):
            diffs.extend(dict_diff(a, e, current_path))
        elif a != e:
            diffs.append(f"{current_path}: actual={a!r}, expected={e!r}")

    return diffs


@pytest.fixture(autouse=True)
def reset_config():
    cm._global_cfg = None


def test_parse_arguments_returns_launch_args():
    result = parse_arguments(argv=[""])

    assert isinstance(result, LaunchArgs)


def test_default_arguments():
    args = parse_arguments(argv=[""])

    excluded = {"_args_raw", "control_type", "learning_method", "debug_record_dir"}
    for f in fields(args):
        if f.name not in excluded:
            assert getattr(args, f.name) in (None, False), (
                f"Expected '{f.name}' to be None or False by default, got {getattr(args, f.name)!r}"
            )

    assert args.control_type == Control.SIM
    assert args.learning_method == Stage.RL
    assert str(args.debug_record_dir).startswith("/tmp/")


def test_get_config_before_setup_raises():
    with pytest.raises(AttributeError):
        cm.get_config()


def test_setup_config():
    cm.setup_config(argv=[""])


def test_setup_config_twice_raises():
    cm.setup_config(argv=[""])

    with pytest.raises(AttributeError):
        cm.setup_config(argv=[""])


def test_get_config():
    cm.setup_config(argv=[""])

    config = cm.get_config()
    assert config is not None


def test_env_config_image_resolution():
    cm.setup_config(argv=[""])
    val = cm.get_config().env_cfg.image_resolution

    assert isinstance(val, tuple)
    assert len(val) == 2
    assert isinstance(val[0], int)
    assert isinstance(val[1], int)


def test_robot_config_parsing():
    cm.setup_config(argv=[""])
    cfg = cm.get_config()

    env_dict = cfg.robot_cfg.as_dict()
    def_dict = cm._get_default_config_dict()["robot"]

    diffs = dict_diff(env_dict, def_dict)
    assert not diffs, "\n".join(diffs)
