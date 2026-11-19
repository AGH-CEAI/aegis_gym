# Peg-in-hole assets

The peg and the board with the hole, as URDF (collision from box primitives, visual from the STL) and STL (visual,
3D printing). Like the Push-T tee in `../push_t/`, the envs load the files from this folder.

| File | What |
|---|---|
| `nominal.yaml` | **The only file to edit**: the dimensions, in metres and kilograms |
| `peg.urdf`, `peg.stl` | the peg: 25 x 25 x 80 mm, 40 g |
| `board.urdf`, `board.stl` | the board: 30 mm thick, one 26 mm hole (1 mm clearance), 5 mm floor |
| `params.yaml` | the parameters the files were generated from |

The URDF, STL and `params.yaml` files are generated: do not edit them by hand.

## Frames

- **Peg:** origin at the centre of its bottom face (the tip), z along its axis, up. Standing on a table top at height
  `z_table`, its pose is `(x, y, z_table, quat)`.
- **Board:** origin at the centre of its top face, z up; the hole inlet is at `(0, 0, 0)` of the board. Standing on the
  table, its pose is `(x, y, z_table + thickness, quat)`.
- The grasp point of the peg (`peg.grasp_from_top` below its top face) and the hole frames are available from the
  generator: `build_geometry(PihAssetsCfg.from_yaml())` gives `peg_grasp_frame` and `slots[i].pose`.

## Changing the assets

1. Edit `nominal.yaml`. For example:
   - another hole: `board.clearances: [0.0005]`;
   - several holes side by side: `board.clearances: [0.005, 0.001, 0.0005]`, spaced by `board.slot_pitch`;
   - a peg with several legs: `peg.legs.grid: [2, 1]`, `peg.legs.pitch`, `peg.legs.length`.

   The config is validated when it loads, so an impossible geometry (overlapping holes, no wall between legs) gives a
   clear error.
2. Regenerate the files, from the repo root in the toolbox (the host Python lacks the dependencies):

   ```bash
   toolbox enter aegis_gym_dev-v0.1.0
   cd ~/ceai_ws/src/aegis_gym
   python3 -m scripts.peg_in_hole.generate_assets
   ```

   It writes `peg.urdf`, `peg.stl`, `board.urdf`, `board.stl` and `params.yaml` here. The generation is deterministic:
   unchanged parameters give byte-identical files.
3. Commit `nominal.yaml` together with the regenerated files. `test/pih_assets_export_test.py` fails while the files
   here do not match `nominal.yaml`.

A variant for an experiment, without touching the files here:

```bash
python3 -m scripts.peg_in_hole.generate_assets --out-dir /tmp/pih_05 --overrides '{"board": {"clearances": [0.0005]}}'
```

From Python, the same with `export_pih_assets(cfg, out_dir)`, or `generate_pih_assets(cfg)` to a cache directory named
by a hash of the parameters (`~/.cache/aegis_gym/pih_assets/`, or `$AEGIS_GYM_PIH_CACHE`):

```python
from aegis_gym.aux.pih_assets import PihAssetsCfg, generate_pih_assets

assets = generate_pih_assets(
    PihAssetsCfg.from_yaml(overrides={"board": {"clearances": [0.0005]}})
)
assets.peg_urdf, assets.board_urdf  # paths to load
```

## Loading in an env

As the tee in `PushTEnv._setup_scene()`: after `scene.add_manipulator()`, before `scene.build()`.

```python
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "peg_in_hole"

self.board = self._scene.add_entity(
    entity=ObjectType.URDF,
    properties=ObjectProperties(
        dims=(),
        pose=(0.47, 0.0, table_top_z + 0.03, 1.0, 0.0, 0.0, 0.0),
        fixed=True,
        color=(0.25, 0.25, 0.3),
        urdf_path=str(ASSETS_DIR / "board.urdf"),
    ),
)
self.peg = self._scene.add_entity(
    entity=ObjectType.URDF,
    properties=ObjectProperties(
        dims=(),
        pose=(0.40, 0.10, table_top_z, 1.0, 0.0, 0.0, 0.0),
        fixed=False,
        color=(0.85, 0.45, 0.1),
        urdf_path=str(ASSETS_DIR / "peg.urdf"),
    ),
)
```

`table_top_z = table_size[2] - workbench_size[2]` from `config/defaults/env.yaml`.

## Example scene

`scripts/peg_in_hole/robot_scene_example.py` builds the robot scene with the board on the table and the peg standing
next to it from these files, simulates 2 s and checks that both stand where placed. In the toolbox, from the repo root
(the robot URDF comes from the local ClearML dataset cache, no server needed; `--urdf PATH` gives it explicitly):

```bash
python3 -m scripts.peg_in_hole.robot_scene_example --vis
python3 -m scripts.peg_in_hole.robot_scene_example --positions '{"board_xy": [0.5, 0.1], "board_yaw_deg": 30}'
```
