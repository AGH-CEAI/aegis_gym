"""Peg-in-hole robot scene placement, without Genesis (a stub scene and manipulator)."""

import math

import pytest
import torch as th

from aegis_gym.aux.geom.geom import transform_by_quat
from aegis_gym.aux.pih_assets import PihAssetsCfg, generate_pih_assets
from aegis_gym.aux.pih_assets.scene import (
    PihSceneCfg,
    add_pih_entities,
    grasp_peg,
    grasp_slip,
    grasp_width,
    hole_poses,
    peg_pose_in_hand,
)
from aegis_gym.envs.objects import ObjectType

TABLE_Z = -0.003
DOWN = (0.0, 1.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def assets(tmp_path_factory):
    return generate_pih_assets(
        PihAssetsCfg.from_yaml(), cache_dir=tmp_path_factory.mktemp("pih")
    )


def _axis(quat: th.Tensor, axis: int) -> th.Tensor:
    unit = th.zeros(len(quat), 3)
    unit[:, axis] = 1.0
    return transform_by_quat(unit, quat)


class _Object:
    def __init__(self, entity, properties):
        self.entity, self.properties = entity, properties
        self.pose = th.tensor(properties.pose, dtype=th.float32).repeat(2, 1)

    def get_pose(self):
        return self.pose.clone()

    def set_pose(self, pose, envs_idx):
        self.pose[envs_idx] = pose


class _Scene:
    def __init__(self):
        self.added = []

    def add_entity(self, entity, properties):
        self.added.append(_Object(entity, properties))
        return self.added[-1]


class _Manipulator:
    """Records the reset calls; the TCP ends exactly at the requested pose."""

    device = th.device("cpu")

    def __init__(self):
        self.calls = []
        self.tcp = th.tensor([[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)

    def ctrl_go_to_home(self, envs_idx):
        self.calls.append("home")

    def ctrl_reset_to_pose(self, pose, open_gripper, envs_idx):
        self.calls.append(("reset", open_gripper))
        self.tcp[envs_idx] = pose

    def set_gripper_width(self, width, envs_idx):
        self.calls.append(("width", round(width, 6)))

    def weld_object(self, obj):
        self.calls.append(("weld", obj))

    def get_tcp_pose(self):
        return self.tcp.clone()


def test_cfg_defaults_and_overrides():
    assert PihSceneCfg.from_dict(None) == PihSceneCfg()
    cfg = PihSceneCfg.from_dict({"board_xy": [0.5, 0.1], "board_yaw_deg": 30})
    assert cfg.board_xy == (0.5, 0.1) and cfg.board_yaw_deg == 30
    assert PihSceneCfg.from_dict(cfg.to_dict()) == cfg


def test_cfg_rejects_unknown_keys_and_bad_lengths():
    with pytest.raises(ValueError, match="board_pos"):
        PihSceneCfg.from_dict({"board_pos": [0.5, 0.0]})
    with pytest.raises(ValueError, match="tcp_start"):
        PihSceneCfg.from_dict({"tcp_start": [0.4, 0.0]})
    with pytest.raises(ValueError, match="grasp"):
        PihSceneCfg.from_dict({"grasp": "glue"})


def test_grasp_width_is_the_peg_side(assets):
    assert grasp_width(assets) == pytest.approx(assets.cfg.peg.side)


def test_a_multileg_peg_is_grasped_by_its_base(tmp_path):
    overrides = {
        "peg": {"legs": {"grid": [2, 1], "length": 0.03}},
        "board": {"slot_pitch": 0.1},
    }
    cfg = PihAssetsCfg.from_yaml(overrides=overrides)
    multi = generate_pih_assets(cfg, cache_dir=tmp_path)
    assert grasp_width(multi) == pytest.approx(cfg.peg.legs.pitch + cfg.peg.side)
    # the nominal pitch makes a 65 mm base, wider than the Hand-E opening
    with pytest.raises(ValueError, match="Hand-E"):
        add_pih_entities(
            _Scene(), PihSceneCfg(assets=overrides), TABLE_Z, cache_dir=str(tmp_path)
        )


@pytest.mark.parametrize("peg_yaw_deg", [0.0, 30.0])
def test_peg_hangs_from_the_pads_along_the_tool_axis(assets, peg_yaw_deg):
    cfg = PihSceneCfg(peg_yaw_deg=peg_yaw_deg)
    tilt = (math.cos(0.2), 0.0, math.sin(0.2), 0.0)  # any TCP orientation
    tcp = th.tensor([[0.4, 0.1, 0.2, *DOWN], [0.3, -0.1, 0.1, *tilt]])
    peg = peg_pose_in_hand(tcp, assets, cfg)

    tool_z, peg_z = _axis(tcp[:, 3:], 2), _axis(peg[:, 3:], 2)
    assert th.allclose(peg_z, -tool_z, atol=1e-6)  # the tip points out of the hand

    pads = tcp[:, :3] - cfg.pad_offset * tool_z
    grasp = peg[:, :3] + transform_by_quat(
        th.tensor(assets.peg_grasp_frame[:3]).expand(2, 3), peg[:, 3:]
    )
    assert th.allclose(grasp, pads, atol=1e-6)

    # the peg x (its face normal) is the tool x (the finger axis) turned by the peg yaw
    cos = th.sum(_axis(peg[:, 3:], 0) * _axis(tcp[:, 3:], 0), dim=1)
    assert th.allclose(
        cos, th.full((2,), math.cos(math.radians(peg_yaw_deg))), atol=1e-6
    )


def test_hole_poses_follow_the_board(assets):
    cfg = PihSceneCfg(board_xy=(0.5, 0.1), board_yaw_deg=90.0)
    holes = hole_poses(cfg, assets, TABLE_Z)
    top = TABLE_Z + assets.cfg.board.thickness
    for frame, pose in zip(assets.hole_frames, holes, strict=True):
        x, y, z = frame.pose[:3]
        # a quarter turn: the board x runs along the world y
        expected = th.tensor([0.5 - y, 0.1 + x, top + z])
        assert th.allclose(pose[:3], expected, atol=1e-6)
        assert th.allclose(
            _axis(pose[None, 3:], 2)[0], th.tensor([0.0, 0.0, 1.0]), atol=1e-6
        )


def test_add_entities_places_the_board_on_the_table_and_the_peg_in_hand(
    assets, tmp_path
):
    scene = _Scene()
    cfg = PihSceneCfg(board_xy=(0.5, 0.0), peg_friction=1.0)
    pih = add_pih_entities(scene, cfg, TABLE_Z, cache_dir=str(tmp_path))

    board, peg = scene.added
    assert board.entity == peg.entity == ObjectType.URDF
    assert board.properties.fixed and not peg.properties.fixed
    assert board.properties.urdf_path == str(pih.assets.board_urdf)
    assert peg.properties.friction == 1.0 and board.properties.friction is None
    assert board.properties.pose[:3] == pytest.approx(
        (0.5, 0.0, TABLE_Z + assets.cfg.board.thickness)
    )
    start = th.tensor(pih.tcp_start_pose)
    assert pih.tcp_start_pose[2] == pytest.approx(TABLE_Z + cfg.tcp_start[2])
    assert th.allclose(
        th.tensor(peg.properties.pose),
        peg_pose_in_hand(start, assets, cfg)[0],
        atol=1e-6,
    )


def test_grasp_resets_the_tcp_and_puts_the_peg_in_hand(tmp_path):
    scene, manipulator = _Scene(), _Manipulator()
    pih = add_pih_entities(scene, PihSceneCfg(), TABLE_Z, cache_dir=str(tmp_path))
    pih.peg.pose[:] = th.tensor(
        [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    )  # where the last episode left it

    grasp_peg(manipulator, pih, th.tensor([1]))

    assert manipulator.calls == ["home", ("reset", False), ("width", 0.025)]
    slip = grasp_slip(manipulator, pih)
    assert slip[1].abs().max() < 1e-6  # the reset env holds the peg
    assert slip[0, 0] > 1.0  # the other env was left alone


def test_a_welded_grasp_also_welds_the_peg_to_the_hand(tmp_path):
    scene, manipulator = _Scene(), _Manipulator()
    cfg = PihSceneCfg(grasp="weld")
    pih = add_pih_entities(scene, cfg, TABLE_Z, cache_dir=str(tmp_path))

    grasp_peg(manipulator, pih, th.tensor([0, 1]))

    assert manipulator.calls[-1] == ("weld", pih.peg)
    assert grasp_slip(manipulator, pih).abs().max() < 1e-6
