"""
Peg drops in Genesis (skipped without it, e.g. in CI): one scene, one env per deterministic scenario and slot.

The drop runs in a subprocess: `gs.init()` switches torch's default device to CUDA for the whole process, which
would break the other tests that run on CPU stubs.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("genesis")

_SCENARIOS = ("centred", "rim", "edge-tilt")
# a fall from ~4.5 cm moves ~3.7 mm per 4 ms step; deeper would mean tunnelling
_IMPACT_LIMIT_MM = 5.0

_RUN = textwrap.dedent(
    """
    import json, sys
    import genesis as gs
    import torch as th
    from aegis_gym.aux.pih_assets import PihAssetsCfg, generate_pih_assets
    from aegis_gym.aux.pih_assets.drop_sim import DropCfg, DropSim, scenario_starts, stack_starts

    cache_dir, names = sys.argv[1], sys.argv[2].split(",")
    gs.init(backend=gs.gpu if th.cuda.is_available() else gs.cpu, precision="32", logging_level="warning")
    assets = generate_pih_assets(PihAssetsCfg.from_yaml(), cache_dir=cache_dir)
    starts = stack_starts(*(scenario_starts(assets, name, n_per_slot=1) for name in names))
    result = DropSim(assets, DropCfg(), starts).run()
    print("RESULT " + json.dumps({"rows": result.rows, "failures": result.failures()}))
    """
)


@pytest.fixture(scope="module")
def result(tmp_path_factory) -> dict:
    env = {**os.environ, "CLEARML_OFFLINE_MODE": "1"}
    cache_dir = str(tmp_path_factory.mktemp("pih"))
    proc = subprocess.run(
        [sys.executable, "-c", _RUN, cache_dir, ",".join(_SCENARIOS)],
        check=False,  # the output is shown on failure below
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    assert proc.returncode == 0 and lines, proc.stdout[-2000:] + proc.stderr[-2000:]
    return json.loads(lines[-1].removeprefix("RESULT "))


def _rows(result: dict, scenario: str) -> list[dict]:
    return [r for r in result["rows"] if r["scenario"] == scenario]


def test_no_physics_errors(result):
    assert result["failures"] == []


def test_centred_pegs_end_in_the_hole(result):
    assert [r["outcome"] for r in _rows(result, "centred")] == ["in_hole"] * 3


def test_pegs_landing_on_the_rim_stay_on_the_board(result):
    assert [r["outcome"] for r in _rows(result, "rim")] == ["on_board"] * 3


def test_pegs_hitting_the_edge_come_to_rest(result):
    # an 80 mm peg hitting the rim tilted topples: onto the board, or off it next to the 5 mm hole
    assert all(r["outcome"] != "moving" for r in _rows(result, "edge-tilt"))


def test_impacts_do_not_tunnel(result):
    assert max(r["transient_penetration_mm"] for r in result["rows"]) < _IMPACT_LIMIT_MM
    assert max(r["rest_penetration_mm"] for r in result["rows"]) < 0.1
