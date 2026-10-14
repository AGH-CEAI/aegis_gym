from types import SimpleNamespace

import pytest

# the wrappers package imports cv2, provided by the container's system packages
pytest.importorskip("cv2")

from aegis_gym.envs.wrappers.base_wrapper import BaseEnvWrapper
from aegis_gym.envs.wrappers.obs_preview_wrapper import ObsPreviewEnvWrapper
from aegis_gym.envs.wrappers.vision_aug_wrapper import VisionAugEnvWrapper


@pytest.mark.parametrize(
    "wrapper_cls", [BaseEnvWrapper, VisionAugEnvWrapper, ObsPreviewEnvWrapper]
)
def test_env_wrappers_are_instantiable(wrapper_cls: type):
    # an abstract method of `BaseEnv` left out by the wrappers fails their instantiation
    assert not wrapper_cls.__abstractmethods__


def test_env_wrapper_delegates_the_default_env_specific_dict():
    env = SimpleNamespace(
        get_default_env_specific_dict=lambda: {"friction": 0.4},
        _scene=None,  # checked by `BaseEnv.__del__()` of the wrapper
    )

    wrapper = BaseEnvWrapper(env=env)

    assert wrapper.get_default_env_specific_dict() == {"friction": 0.4}
