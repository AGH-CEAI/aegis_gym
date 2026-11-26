import torch as th

import aegis_gym.config  # noqa: F401
from aegis_gym.config.types import DomainRandomizationCfg, ObjectColorCfg
from aegis_gym.envs.scene.sim.genesis_scene import GenesisScene

NUM_ENVS = 256
BLACK, GREEN = [0.02, 0.02, 0.02], [0.1, 0.6, 0.15]


def _scene(object_color: ObjectColorCfg) -> GenesisScene:
    """The colour state of `enable_object_masks()` without a Genesis scene."""
    scene = object.__new__(GenesisScene)
    scene.device = th.device("cpu")
    scene._cfg_dr = DomainRandomizationCfg(enabled=True, object_color=object_color)
    scene._object_nominal_color = th.tensor(GREEN)
    scene._object_colors = scene._object_nominal_color.repeat(NUM_ENVS, 1)
    return scene


def test_the_palette_colours_are_drawn_per_env():
    scene = _scene(ObjectColorCfg(enabled=True, palette=[BLACK, GREEN]))

    scene._rand_object_color(th.arange(NUM_ENVS))

    colors = scene._object_colors
    is_black = th.isclose(colors, th.tensor(BLACK)).all(dim=1)
    is_green = th.isclose(colors, th.tensor(GREEN)).all(dim=1)
    assert (is_black | is_green).all()
    assert 0.3 < is_black.float().mean() < 0.7


def test_off_the_palette_the_colours_are_uniform():
    scene = _scene(ObjectColorCfg(enabled=True, palette=[BLACK], palette_prob=0.0))

    scene._rand_object_color(th.arange(NUM_ENVS))

    assert not th.isclose(scene._object_colors, th.tensor(BLACK)).all(dim=1).any()


def test_only_the_given_envs_are_redrawn_and_disabled_keeps_the_nominal():
    on = _scene(ObjectColorCfg(enabled=True, palette=[BLACK]))
    off = _scene(ObjectColorCfg(enabled=False, palette=[BLACK]))

    on._rand_object_color(th.tensor([0, 1]))
    off._rand_object_color(th.arange(NUM_ENVS))

    assert th.equal(on._object_colors[:2], th.tensor([BLACK, BLACK]))
    assert th.equal(on._object_colors[2:], th.tensor(GREEN).repeat(NUM_ENVS - 2, 1))
    assert th.equal(off._object_colors, th.tensor(GREEN).repeat(NUM_ENVS, 1))
