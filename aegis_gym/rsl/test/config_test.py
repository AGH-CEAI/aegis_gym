import pytest

from config import LaunchArgs, parse_arguments
from config import ConfigManager as cm


@pytest.fixture(autouse=True)
def reset_config():
    cm._global_cfg = None


def test_parse_arguments_returns_launch_args():
    result = parse_arguments(argv=[""], extra_argparser=None)

    assert isinstance(result, LaunchArgs)


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
