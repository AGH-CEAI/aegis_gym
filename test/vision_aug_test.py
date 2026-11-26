from types import SimpleNamespace

import pytest
import torch as th
from tensordict import TensorDict

import aegis_gym.config  # noqa: F401
from aegis_gym.config.types import IMAGE_MODALITIES, CutoutCfg, ImageAugCfg
from aegis_gym.envs.wrappers import VisionAugEnvWrapper

NUM_ENVS = 64
RES = (16, 16)


def _no_photometric(**cutout) -> ImageAugCfg:
    """Only the cut-out, the other augmentations off."""
    return ImageAugCfg(
        enabled=True,
        per_episode_aug=True,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
        gaussian_noise_std=0.0,
        gamma_range=0.0,
        channel_jitter=0.0,
        blur_prob=0.0,
        cutout=CutoutCfg(**cutout),
    )


def _wrapper(cfg: ImageAugCfg) -> VisionAugEnvWrapper:
    env = SimpleNamespace(num_envs=NUM_ENVS, device=th.device("cpu"), _scene=None)
    return VisionAugEnvWrapper(
        env=env, cfg_image_aug=cfg, cfg_env=SimpleNamespace(image_resolution=RES)
    )


def _images(value: float = 0.5) -> TensorDict:
    return TensorDict(
        {m.value: th.full((NUM_ENVS, 3, *RES), value) for m in IMAGE_MODALITIES},
        batch_size=[NUM_ENVS],
    )


def test_cutout_boxes_are_black_and_stay_within_the_drawn_sizes():
    wrapper = _wrapper(_no_photometric(prob=1.0, min_size=3, max_size=5))

    out = wrapper._augment(_images())

    covered = (out[IMAGE_MODALITIES[0].value] == 0.0).all(dim=1).sum(dim=(1, 2))
    assert ((covered >= 9) & (covered <= 25)).all()


def test_cutout_is_off_with_prob_0_and_on_in_every_env_with_prob_1():
    off = _wrapper(_no_photometric(prob=0.0, max_size=8))._augment(_images())
    on = _wrapper(_no_photometric(prob=1.0, max_size=8))._augment(_images())

    assert (off[IMAGE_MODALITIES[0].value] == 0.5).all()
    assert (on[IMAGE_MODALITIES[0].value] == 0.0).flatten(1).any(dim=1).all()


def test_boxes_larger_than_half_the_image_fit_in_it():
    wrapper = _wrapper(_no_photometric(prob=1.0, min_size=12, max_size=16))

    out = wrapper._augment(_images())

    covered = (
        (out[IMAGE_MODALITIES[0].value] == 0.0).all(dim=1).float().mean(dim=(1, 2))
    )
    assert (covered >= 144 / 256).all()


def test_several_boxes_per_image():
    one = _wrapper(_no_photometric(prob=1.0, min_size=2, max_size=2, count=1))._augment(
        _images()
    )
    many = _wrapper(
        _no_photometric(prob=1.0, min_size=2, max_size=2, count=6)
    )._augment(_images())

    def covered(out):
        return (
            (out[IMAGE_MODALITIES[0].value] == 0.0)
            .all(dim=1)
            .sum(dim=(1, 2))
            .float()
            .mean()
        )

    assert covered(many) > 2 * covered(one)


def test_the_boxes_are_the_same_on_all_cameras_unless_per_camera():
    shared = _wrapper(_no_photometric(prob=1.0, max_size=8))._augment(_images())
    own = _wrapper(_no_photometric(prob=1.0, max_size=8, per_camera=True))._augment(
        _images()
    )

    first, second = (m.value for m in IMAGE_MODALITIES[:2])
    assert th.equal(shared[first], shared[second])
    assert not th.equal(own[first], own[second])


def test_the_boxes_last_the_episode_until_resampled():
    wrapper = _wrapper(_no_photometric(prob=1.0, max_size=8))
    key = IMAGE_MODALITIES[0].value

    first = wrapper._augment(_images())[key]
    again = wrapper._augment(_images())[key]
    wrapper.resample_augmentation()
    redrawn = wrapper._augment(_images())[key]

    assert th.equal(first, again)
    assert not th.equal(first, redrawn)


def test_color_fill_paints_a_solid_colour_per_box():
    wrapper = _wrapper(_no_photometric(prob=1.0, min_size=4, max_size=4, fill="color"))

    out = wrapper._augment(_images())[IMAGE_MODALITIES[0].value]

    changed = (out != 0.5).any(dim=1)  # [N, H, W]
    colors = wrapper._aug_profile["cutout_color"][:, 0, 0]  # [N, 3]
    for i in range(NUM_ENVS):
        assert th.allclose(
            out[i][:, changed[i]], colors[i][:, None].expand(3, int(changed[i].sum()))
        )


def test_noise_fill_changes_every_frame():
    wrapper = _wrapper(_no_photometric(prob=1.0, min_size=8, max_size=8, fill="noise"))
    key = IMAGE_MODALITIES[0].value

    first, second = wrapper._augment(_images())[key], wrapper._augment(_images())[key]

    assert not th.equal(first, second)
    assert th.equal(first == 0.5, second == 0.5)  # the same boxes


def test_unknown_fill_is_rejected():
    wrapper = _wrapper(_no_photometric(prob=1.0, fill="checkerboard"))

    with pytest.raises(ValueError, match="checkerboard"):
        wrapper._augment(_images())


def test_an_auto_reset_episode_gets_new_boxes_the_others_keep_theirs():
    wrapper = _wrapper(_no_photometric(prob=1.0, max_size=8))
    dones = th.zeros(NUM_ENVS, dtype=th.bool)
    dones[0] = True
    wrapper._env.step = lambda actions: (None, None, dones, {})
    before = {k: v.clone() for k, v in wrapper._aug_profile.items()}

    wrapper.step(th.zeros(NUM_ENVS, 2))

    corners = ("cutout_y", "cutout_x", "cutout_h", "cutout_w")
    assert any(not th.equal(before[k][0], wrapper._aug_profile[k][0]) for k in corners)
    assert all(th.equal(before[k][1:], wrapper._aug_profile[k][1:]) for k in before)
