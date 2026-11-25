import itertools

import pytest

from aegis_gym.aux.pih_assets import (
    Box,
    PihAssetsCfg,
    Rect,
    build_geometry,
    decompose_plate,
)

_TOL = 1e-9
# nominal board holds the tolerance family; geometry is checked on three slots side by side
_THREE_SLOTS = {"board": {"clearances": [0.005, 0.001, 0.0005]}}
_TWO_LEGS = {
    "peg": {"legs": {"grid": [2, 1], "length": 0.03}},
    "board": {"slot_pitch": 0.1, **_THREE_SLOTS["board"]},
}
_MULTILEG = {
    "peg": {"legs": {"grid": [2, 2], "pitch": 0.04, "length": 0.03}},
    "board": {"slot_pitch": 0.1, **_THREE_SLOTS["board"]},
}


def _cfg(overrides: dict | None = None) -> PihAssetsCfg:
    return PihAssetsCfg.from_yaml(overrides=overrides)


def _overlap(a: Box, b: Box) -> bool:
    (also, ahi), (blo, bhi) = a.bounds, b.bounds
    return all(min(ahi[i], bhi[i]) - max(also[i], blo[i]) > _TOL for i in range(3))


def _hole_box(h: Rect, depth: float) -> Box:
    return Box(
        center=((h.x0 + h.x1) / 2, (h.y0 + h.y1) / 2, -depth / 2),
        size=(h.x1 - h.x0, h.y1 - h.y0, depth),
    )


@pytest.fixture(params=[_THREE_SLOTS, _TWO_LEGS, _MULTILEG], ids=["1x1", "2x1", "2x2"])
def cfg(request) -> PihAssetsCfg:
    return _cfg(request.param)


def test_nominal_is_the_single_leg_tolerance_family():
    cfg = _cfg()
    assert cfg.board.clearances == (0.004, 0.003, 0.002, 0.001)
    assert cfg.peg.legs.grid == (1, 1)


def test_hole_side_matches_clearance(cfg):
    geom = build_geometry(cfg)
    assert [s.clearance for s in geom.slots] == list(cfg.board.clearances)
    for slot in geom.slots:
        assert len(slot.holes) == cfg.peg.legs.count
        for h in slot.holes:
            assert h.x1 - h.x0 - cfg.peg.side == pytest.approx(slot.clearance, abs=_TOL)
            assert h.y1 - h.y0 - cfg.peg.side == pytest.approx(slot.clearance, abs=_TOL)


def test_holes_centred_on_legs(cfg):
    """With the peg frame on a slot frame, every leg sits in the centre of its hole."""
    geom = build_geometry(cfg)
    legs = [b for b in geom.peg_boxes if b.center[2] < cfg.peg.leg_length]
    for slot in geom.slots:
        hole_centres = sorted(
            ((h.x0 + h.x1) / 2 - slot.pose[0], (h.y0 + h.y1) / 2 - slot.pose[1])
            for h in slot.holes
        )
        leg_centres = sorted(b.center[:2] for b in legs)
        for hc, lc in zip(hole_centres, leg_centres, strict=True):
            assert hc == pytest.approx(lc, abs=_TOL)


def test_board_boxes_disjoint_and_clear_of_holes(cfg):
    geom = build_geometry(cfg)
    depth = cfg.board.depth
    holes = [_hole_box(h, depth) for s in geom.slots for h in s.holes]
    for a, b in itertools.combinations(geom.board_boxes, 2):
        assert not _overlap(a, b)
    for box, hole in itertools.product(geom.board_boxes, holes):
        assert not _overlap(box, hole)


def test_board_boxes_inside_footprint(cfg):
    geom = build_geometry(cfg)
    fp, t = geom.board_footprint, cfg.board.thickness
    for box in geom.board_boxes:
        lo, hi = box.bounds
        assert lo[0] >= fp.x0 - _TOL and hi[0] <= fp.x1 + _TOL
        assert lo[1] >= fp.y0 - _TOL and hi[1] <= fp.y1 + _TOL
        assert lo[2] >= -t - _TOL and hi[2] <= _TOL


def test_board_volume_is_plate_minus_holes(cfg):
    """Disjoint boxes inside the footprint with this volume cover the board exactly."""
    geom = build_geometry(cfg)
    fp = geom.board_footprint
    plate = (fp.x1 - fp.x0) * (fp.y1 - fp.y0) * cfg.board.thickness
    holes = (
        sum((h.x1 - h.x0) * (h.y1 - h.y0) for s in geom.slots for h in s.holes)
        * cfg.board.depth
    )
    assert sum(b.volume for b in geom.board_boxes) == pytest.approx(
        plate - holes, rel=1e-12
    )


def test_margin_around_holes(cfg):
    geom = build_geometry(cfg)
    holes = [h for s in geom.slots for h in s.holes]
    fp = geom.board_footprint
    assert min(h.x0 for h in holes) - fp.x0 == pytest.approx(cfg.board.margin, abs=_TOL)
    assert fp.y1 - max(h.y1 for h in holes) == pytest.approx(cfg.board.margin, abs=_TOL)


def test_peg_dimensions(cfg):
    geom = build_geometry(cfg)
    lows, highs = zip(*(b.bounds for b in geom.peg_boxes), strict=True)
    assert min(lo[2] for lo in lows) == pytest.approx(
        0.0, abs=_TOL
    )  # tip plane is the peg frame
    assert max(hi[2] for hi in highs) == pytest.approx(cfg.peg.length, abs=_TOL)
    legs = [b for b in geom.peg_boxes if b.size[:2] == (cfg.peg.side, cfg.peg.side)]
    assert len(legs) == cfg.peg.legs.count
    for a, b in itertools.combinations(geom.peg_boxes, 2):
        assert not _overlap(a, b)


def test_single_peg_is_one_box():
    geom = build_geometry(_cfg())
    assert len(geom.peg_boxes) == 1
    assert geom.peg_boxes[0].size == pytest.approx((0.025, 0.025, 0.08))


def test_frames():
    cfg = _cfg(_THREE_SLOTS)
    geom = build_geometry(cfg)
    assert geom.peg_tip_frame == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert geom.peg_grasp_frame[2] == pytest.approx(
        cfg.peg.length - cfg.peg.grasp_from_top
    )
    xs = [s.pose[0] for s in geom.slots]
    assert xs == pytest.approx([-cfg.board.slot_pitch, 0.0, cfg.board.slot_pitch])
    assert all(s.pose[2] == 0.0 for s in geom.slots)  # inlets on the top face


def test_through_holes_have_no_floor():
    blind = build_geometry(_cfg(_THREE_SLOTS))
    through = build_geometry(_cfg({"board": {"bottom": 0.0, **_THREE_SLOTS["board"]}}))
    assert len(blind.board_boxes) - len(through.board_boxes) == 3


def test_deterministic(cfg):
    assert build_geometry(cfg) == build_geometry(cfg)


def test_overrides_merge_and_keep_the_rest():
    cfg = _cfg({"board": {"clearances": [0.0002]}})
    assert cfg.board.clearances == (0.0002,)
    assert cfg.board.thickness == _cfg().board.thickness


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "peg": {"legs": {"grid": [2, 1], "length": 0.03, "pitch": 0.026}}
        },  # no wall between holes
        {"peg": {"legs": {"grid": [2, 1]}}},  # multileg needs a leg length
        {"board": {"slot_pitch": 0.02, **_THREE_SLOTS["board"]}},  # slots overlap
        {"board": {"bottom": 0.03}},  # no depth left
        {"board": {"clearances": [-0.001]}},
        {"peg": {"grasp_from_top": 0.1}},
        {"board": {"typo": 1}},
    ],
)
def test_invalid_config_raises(overrides):
    with pytest.raises(ValueError):
        _cfg(overrides)


def test_decompose_single_hole_is_four_boxes():
    rects = decompose_plate(Rect(-1, 1, -1, 1), [Rect(-0.5, 0.5, -0.5, 0.5)])
    assert len(rects) == 4
    assert sum((r.x1 - r.x0) * (r.y1 - r.y0) for r in rects) == pytest.approx(4 - 1)


def test_decompose_merges_strips_with_the_same_free_interval():
    """The free band under the first hole spans every strip that hole covers, as one box."""
    rects = decompose_plate(Rect(0, 5, 0, 5), [Rect(1, 3, 1, 2), Rect(2, 4, 3, 4)])
    assert Rect(1, 3, 0, 1) in rects
    assert sum((r.x1 - r.x0) * (r.y1 - r.y0) for r in rects) == pytest.approx(
        25 - 2 - 2
    )
