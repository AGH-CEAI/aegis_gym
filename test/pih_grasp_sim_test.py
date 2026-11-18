"""
The peg-in-hole robot scene in Genesis (skipped without it, e.g. in CI, or without the robot URDF in the local ClearML
cache): the peg held in the hand while the TCP moves, with the physical and the welded grasp.

Each grasp runs `scripts.peg_in_hole.grasp_test` in a subprocess: `gs.init()` switches torch's default device to CUDA
for the whole process (see `pih_drop_sim_test.py`), and the global config can be set up only once per process.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("genesis")

_REPO = Path(__file__).resolve().parents[1]
_DEFAULTS = _REPO / "aegis_gym" / "config" / "defaults"
_ROBOT_YAML = _DEFAULTS / "robot.yaml"
_DATASETS = Path("~/.clearml/cache/storage_manager/datasets").expanduser()


def _cached_urdf() -> bool:
    urdf_id = yaml.safe_load(_ROBOT_YAML.read_text())["urdf_id_cell"]
    return any((_DATASETS / f"ds_{urdf_id}").rglob("*.urdf"))


pytestmark = pytest.mark.skipif(
    not _cached_urdf(), reason="no robot URDF in the ClearML cache"
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.peg_in_hole.grasp_test", "-B", "2", *args],
        check=False,  # the output is shown on failure below
        capture_output=True,
        text=True,
        timeout=900,
        cwd=_REPO,
        env={**os.environ, "CLEARML_OFFLINE_MODE": "1"},
    )


@pytest.mark.parametrize("grasp", ["physical", "weld"])
def test_the_peg_stays_in_the_hand(grasp):
    proc = _run("--speeds", "1", "--scene", f'{{"grasp": "{grasp}"}}')
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "\nOK\n" in proc.stdout, out[-3000:]
    # the board stands on the table: its top face is the board thickness above the table top
    env = yaml.safe_load((_DEFAULTS / "env.yaml").read_text())
    nominal = yaml.safe_load(
        (_REPO / "aegis_gym" / "assets" / "peg_in_hole" / "nominal.yaml").read_text()
    )
    top = (
        env["table_size"][2] - env["workbench_size"][2] + nominal["board"]["thickness"]
    )
    line = next(r for r in proc.stdout.splitlines() if r.startswith("board top at"))
    board_z = float(line.split("(")[1].split(")")[0].split(",")[2])
    assert board_z == pytest.approx(top)
