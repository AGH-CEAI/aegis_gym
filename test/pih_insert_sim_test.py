"""
Scripted peg insertion in the peg-in-hole robot scene, in Genesis (skipped without it, e.g. in CI, or without the robot
URDF in the local ClearML cache): a centred peg goes into every hole, a peg 3 mm off (over half of every clearance)
stays on the rim.

Runs `scripts.peg_in_hole.insert_test` in a subprocess, as `pih_grasp_sim_test.py` does `grasp_test`.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("genesis")

_REPO = Path(__file__).resolve().parents[1]
_ROBOT_YAML = _REPO / "aegis_gym" / "config" / "defaults" / "robot.yaml"
_DATASETS = Path("~/.clearml/cache/storage_manager/datasets").expanduser()


def _cached_urdf() -> bool:
    urdf_id = yaml.safe_load(_ROBOT_YAML.read_text())["urdf_id_cell"]
    return any((_DATASETS / f"ds_{urdf_id}").rglob("*.urdf"))


pytestmark = pytest.mark.skipif(
    not _cached_urdf(), reason="no robot URDF in the ClearML cache"
)


@pytest.fixture(scope="module")
def rows(tmp_path_factory) -> list[dict]:
    out_dir = tmp_path_factory.mktemp("insert")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.peg_in_hole.insert_test",
            "-B",
            "1",
            "--offsets-mm",
            "0",
            "3",
            "--out-dir",
            str(out_dir),
        ],
        check=False,  # the output is shown on failure below
        capture_output=True,
        text=True,
        timeout=900,
        cwd=_REPO,
        env={**os.environ, "CLEARML_OFFLINE_MODE": "1"},
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0 and "\nOK\n" in proc.stdout, out[-3000:]
    (path,) = out_dir.glob("insert_*.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def test_a_centred_peg_goes_into_every_hole(rows):
    centred = [r for r in rows if float(r["offset_mm"]) == 0]
    assert len(centred) == 3
    assert all(r["outcome"] == "inserted" for r in centred)
    assert all(float(r["tip_above_floor_mm"]) <= 1.0 for r in centred)


def test_a_peg_off_by_more_than_half_the_clearance_rests_on_the_rim(rows):
    off = [r for r in rows if float(r["offset_mm"]) == 3]
    assert len(off) == 3
    assert all(r["outcome"] == "blocked" for r in off)


def test_the_peg_does_not_pass_through_the_board(rows):
    assert max(float(r["penetration_max_mm"]) for r in rows) < 0.5
