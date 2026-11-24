from types import SimpleNamespace

import pytest
import torch as th

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


def test_env_wrapper_delegates_the_state_obs():
    env = SimpleNamespace(
        available_state_obs=frozenset({"tcp_pose"}),
        get_state_obs=lambda names: names,
        _scene=None,
    )
    inner = BaseEnvWrapper(env=env)

    wrapper = BaseEnvWrapper(env=inner)

    assert wrapper.available_state_obs == {"tcp_pose"}
    assert wrapper.get_state_obs(["tcp_pose"]) == ["tcp_pose"]


def test_env_wrapper_delegates_the_seeded_reset_support():
    env = SimpleNamespace(supports_seeded_reset=False, _scene=None)

    assert BaseEnvWrapper(env=BaseEnvWrapper(env=env)).supports_seeded_reset is False


def test_vision_aug_wrapper_redraws_its_augmentation_with_the_domain_randomization():
    calls = []
    env = SimpleNamespace(
        apply_domain_randomization=lambda envs_idx: calls.append(
            ("scene", envs_idx.tolist())
        ),
        _scene=None,
    )
    aug = object.__new__(VisionAugEnvWrapper)
    aug._env = env
    aug._resample_aug_profile = lambda envs_idx: calls.append(
        ("aug", envs_idx.tolist())
    )

    BaseEnvWrapper(env=aug).apply_domain_randomization(th.tensor([0, 2]))

    assert calls == [("scene", [0, 2]), ("aug", [0, 2])]
