import math

import pytest
import torch as th

from aegis_gym.aux.geom import transform_quat_by_quat
from aegis_gym.aux.rand import hash_uniform, rand_uniform
from aegis_gym.config.types import POLICY_PREVIEW_SEEDS
from aegis_gym.envs import PushTEnv, ReacherEnv

# The draws of the `POLICY_PREVIEW_SEEDS`, scaled by 2^24. They define the policy preview scenes:
# a change here means that the previews are no longer comparable with the earlier runs.
PINNED_PREVIEW_DRAWS = [
    [130277, 325842, 1018909],
    [11672718, 8226555, 5889389],
    [14671906, 13280117, 7804980],
    [7055940, 3665557, 10064269],
    [3459856, 8455166, 9777000],
    [2653942, 8558946, 12832761],
    [11462456, 14452853, 2076717],
    [13554128, 8426441, 4286713],
    [9765726, 7922590, 16205737],
    [13198792, 15593700, 8832350],
]

DEVICES = ["cpu"] + (["cuda"] if th.cuda.is_available() else [])


def _pinned_draws() -> th.Tensor:
    return th.tensor(PINNED_PREVIEW_DRAWS, dtype=th.float32) / float(1 << 24)


def _reference_hash_uniform(seed: int, draw: int) -> float:
    """Plain Python reference of `hash_uniform()`, with the arbitrary precision ints."""

    def lowbias32(x: int) -> int:
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        x = (x * 0x846CA68B) & 0xFFFFFFFF
        return x ^ (x >> 16)

    key = (lowbias32(seed) + (draw + 1) * 0x9E3779B9) & 0xFFFFFFFF
    return (lowbias32(key) >> 8) / float(1 << 24)


@pytest.mark.parametrize("device", DEVICES)
def test_hash_uniform_preview_seeds_are_pinned(device: str):
    u = hash_uniform(seeds=POLICY_PREVIEW_SEEDS, num_draws=3, device=device)
    assert th.equal(u.cpu(), _pinned_draws())


def test_hash_uniform_matches_reference():
    seeds = [0, 1, 7, 12345, 2**31, 2**32 - 1]
    u = hash_uniform(seeds=seeds, num_draws=4, device="cpu")
    expected = [[_reference_hash_uniform(s, j) for j in range(4)] for s in seeds]
    assert u.tolist() == expected


@pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA unavailable")
def test_hash_uniform_is_device_independent():
    seeds = list(range(1000))
    assert th.equal(
        hash_uniform(seeds=seeds, num_draws=5, device="cpu"),
        hash_uniform(seeds=seeds, num_draws=5, device="cuda").cpu(),
    )


def test_hash_uniform_range_and_independence():
    u = hash_uniform(seeds=list(range(10000)), num_draws=3, device="cpu")
    assert u.dtype == th.float32
    assert u.min() >= 0.0 and u.max() < 1.0
    assert th.allclose(u.mean(dim=0), th.full((3,), 0.5), atol=0.01)
    # The draws of a seed are not correlated
    assert th.corrcoef(u.T).fill_diagonal_(0.0).abs().max() < 0.05


def test_hash_uniform_rejects_out_of_range_seeds():
    with pytest.raises(ValueError):
        hash_uniform(seeds=[-1], num_draws=1, device="cpu")
    with pytest.raises(ValueError):
        hash_uniform(seeds=[2**32], num_draws=1, device="cpu")


def test_rand_uniform():
    assert rand_uniform(num=5, num_draws=3, device="cpu").shape == (5, 3)
    assert th.equal(
        rand_uniform(num=2, num_draws=3, device="cpu", seeds=[4, 2]),
        hash_uniform(seeds=[4, 2], num_draws=3, device="cpu"),
    )
    with pytest.raises(ValueError):
        rand_uniform(num=3, num_draws=3, device="cpu", seeds=[1, 2])


def _bare_env(env_cls: type, **attrs) -> object:
    """The env with only the attributes needed by its random pose (no scene)."""
    env = object.__new__(env_cls)
    env.device = th.device("cpu")
    env.spawnbox_xlength = 1.0
    env.spawnbox_ylength = 1.0
    env.spawnbox_xoffset = 0.0
    env.spawnbox_yoffset = 0.0
    for name, value in attrs.items():
        setattr(env, name, value)
    return env


def _yaw_quat(yaw: th.Tensor) -> th.Tensor:
    zeros = th.zeros_like(yaw)
    return th.stack([th.cos(yaw / 2), zeros, zeros, th.sin(yaw / 2)], dim=-1)


def test_push_t_seeded_pose_uses_pinned_draws():
    env = _bare_env(PushTEnv, goal_offset_xy=[0.0, 0.0], table_top_z=0.0)
    pose = env._get_random_tee_pose(
        num_reset=len(POLICY_PREVIEW_SEEDS), seeds=POLICY_PREVIEW_SEEDS
    )
    u = _pinned_draws()
    assert th.allclose(pose[:, 0], u[:, 0])
    assert th.allclose(pose[:, 1], u[:, 1])
    assert th.allclose(pose[:, 3:], _yaw_quat(u[:, 2] * 2 * math.pi))


def test_reacher_seeded_pose_uses_pinned_draws():
    env = _bare_env(
        ReacherEnv,
        table_size=[1.0, 1.0, 0.0],
        workbench_size=[1.0, 1.0, 0.0],
        box_size=[0.1, 0.1, 0.0],
    )
    pose = env._get_random_object_pose(
        num_reset=len(POLICY_PREVIEW_SEEDS), seeds=POLICY_PREVIEW_SEEDS
    )
    u = _pinned_draws()
    assert th.allclose(pose[:, 0], u[:, 0])
    assert th.allclose(pose[:, 1], u[:, 1])
    q_downward = th.tensor([0.0, 1.0, 0.0, 0.0]).repeat(len(u), 1)
    q_yaw = _yaw_quat((u[:, 2] * 2 * math.pi - math.pi) * 0.25)
    assert th.allclose(pose[:, 3:], transform_quat_by_quat(q_yaw, q_downward))
