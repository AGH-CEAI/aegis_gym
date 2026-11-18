"""
Writes the peg-in-hole assets (peg and board, URDF and STL, plus `params.yaml`) from `nominal.yaml` into
`aegis_gym/assets/peg_in_hole/`, where the envs load them from, as the Push-T tee. Run it after changing
`nominal.yaml`; `test/pih_assets_export_test.py` fails while the files are out of date.

No Genesis needed. Run from the repo root, e.g.:
    python3 -m scripts.peg_in_hole.generate_assets
    python3 -m scripts.peg_in_hole.generate_assets --out-dir /tmp/pih --overrides '{"board": {"clearances": [0.0005]}}'
"""

import argparse
import json
import sys
from pathlib import Path

from aegis_gym.aux.pih_assets import PihAssetsCfg, export_pih_assets

ASSETS_DIR = (
    Path(__file__).resolve().parents[2] / "aegis_gym" / "assets" / "peg_in_hole"
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out-dir", type=Path, default=ASSETS_DIR)
    p.add_argument(
        "--overrides",
        type=json.loads,
        default=None,
        help="JSON merged into nominal.yaml",
    )
    args = p.parse_args(argv)

    assets = export_pih_assets(
        PihAssetsCfg.from_yaml(overrides=args.overrides), args.out_dir
    )
    for path in (assets.peg_urdf, assets.peg_stl, assets.board_urdf, assets.board_stl):
        print(path)
    print(
        "clearances [mm]: "
        + ", ".join(f"{f.clearance * 1e3:g}" for f in assets.hole_frames)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
