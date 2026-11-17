"""
Batched peg drops onto the generated board in Genesis, without a robot: the contact physics in isolation.

Every parallel environment holds the board and one free peg aimed at one of the slots, as in the task (pegs of
different environments never touch, so a toppling peg cannot lean on another one). While the pegs fall and settle,
the deepest peg-board penetration is tracked every step (`metrics.max_penetration_batch`, the transient penetration
of the impacts) and measured again at rest, and at the end every peg is classified by where it rests.

Scenarios (start poses, relative to the centre of the env's slot):
- `centred`: straight above the hole, upright;
- `rim`: shifted along +x so the peg lands on the board next to the hole, overhanging it (stays on the board);
- `edge-tilt`: shifted so the +x bottom edge is above the rim, tilted 10 deg so that corner hits the rim first;
- `random`: offset uniform in a disc of radius `offset_max` (default clearance / 2 + 2 mm), tilted up to
  `tilt_max` deg about a random horizontal axis.

This module imports Genesis; the rest of `aux.pih_assets` does not depend on it.
"""

import math
import time
from dataclasses import dataclass, field

import genesis as gs
import numpy as np
import torch as th

from aegis_gym.envs.objects import ObjectProperties
from aegis_gym.envs.objects.sim.genesis_objects import GenesisURDF

from .assets import PihAssets
from .metrics import max_penetration_batch, world_vertices

SCENARIOS = ("centred", "rim", "edge-tilt", "random")
OUTCOMES = ("in_hole", "on_board", "fallen", "jammed", "off_board", "moving")

# [m] deeper at rest means the contact let the bodies pass into each other; the transient penetration of an
# impact (about the fall speed times the step) is reported but not judged, insertion is far slower than a drop
PENETRATION_TOL = 5e-4
SEAT_TOL = 1e-3  # [m] lowest peg point to a resting surface
UPRIGHT_TOL = 2.0  # [deg] tilt counted as upright
FALLEN_TILT = 45.0  # [deg] a peg on the board tilted more than this has fallen over
REST_SPEED = 5e-3  # [m/s] a peg slower than this is at rest
JITTER_WINDOW = 0.5  # [s] the end of the drop the jitter is measured over

RIM_OVERHANG = (
    0.003  # [m] how far past the hole edge the peg's near face lands in `rim`
)
EDGE_OVER_RIM = 0.002  # [m] how far the +x bottom edge is above the rim in `edge-tilt`
EDGE_TILT = 10.0  # [deg]

_PEG_COLOR = (0.85, 0.45, 0.1)
_BOARD_COLOR = (0.3, 0.3, 0.38)


@dataclass(frozen=True)
class DropCfg:
    ctrl_dt: float = 0.004  # [s] the rigid solver step, as `env.ctrl_dt`
    # [s] drop duration; a toppling peg needs ~2.5 s to come to rest
    settle: float = 3.0
    hover: float = 0.02  # [m] lowest peg point above the board at the start
    # Genesis RigidOptions overrides, as `env.rigid_contact`
    rigid_contact: dict | None = None


@dataclass
class Starts:
    """Start of every env: target slot, xy offset from the slot centre [m] and orientation (wxyz)."""

    slots: np.ndarray  # (n_envs,) int
    offsets: np.ndarray  # (n_envs, 2)
    quats: np.ndarray  # (n_envs, 4)
    scenarios: list[str]  # (n_envs,)

    @property
    def n_envs(self) -> int:
        return len(self.slots)


def _tilt_quat(tilt_deg: float, axis_angle: float = math.pi / 2) -> np.ndarray:
    """Rotation by `tilt_deg` about the horizontal axis at `axis_angle` from x (pi/2: about y)."""
    half = math.radians(tilt_deg) / 2
    return np.array(
        [
            math.cos(half),
            math.sin(half) * math.cos(axis_angle),
            math.sin(half) * math.sin(axis_angle),
            0.0,
        ]
    )


def scenario_starts(
    assets: PihAssets,
    scenario: str,
    n_per_slot: int,
    seed: int = 0,
    offset_max: float | None = None,
    tilt_max: float = 5.0,
) -> Starts:
    """Start poses of one scenario: `n_per_slot` envs for every slot, slot after slot."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}, expected one of {SCENARIOS}")
    side = assets.cfg.peg.side
    slots = np.repeat(np.arange(len(assets.cfg.board.clearances)), n_per_slot)
    clearance = np.array(assets.cfg.board.clearances)[slots]
    n = len(slots)
    offsets = np.zeros((n, 2))
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    if scenario == "rim":
        # near face RIM_OVERHANG past the hole edge: the peg rests on the board, overhanging the hole
        offsets[:, 0] = (side + clearance) / 2 + RIM_OVERHANG
    elif scenario == "edge-tilt":
        # +x bottom edge EDGE_OVER_RIM over the rim, tilted so that this edge is the lowest
        offsets[:, 0] = clearance / 2 + EDGE_OVER_RIM
        quats[:] = _tilt_quat(EDGE_TILT)
    elif scenario == "random":
        rng = np.random.default_rng(seed)
        radius = clearance / 2 + 0.002 if offset_max is None else np.full(n, offset_max)
        r = radius * np.sqrt(rng.uniform(size=n))
        phi = rng.uniform(0, 2 * math.pi, size=n)
        offsets[:, 0], offsets[:, 1] = r * np.cos(phi), r * np.sin(phi)
        tilts, axes = (
            rng.uniform(0, tilt_max, size=n),
            rng.uniform(0, 2 * math.pi, size=n),
        )
        quats = np.stack([_tilt_quat(t, a) for t, a in zip(tilts, axes, strict=True)])
    return Starts(slots=slots, offsets=offsets, quats=quats, scenarios=[scenario] * n)


def stack_starts(*starts: Starts) -> Starts:
    """Concatenates scenarios along the env axis, e.g. several scenarios in a single scene."""
    return Starts(
        slots=np.concatenate([s.slots for s in starts]),
        offsets=np.concatenate([s.offsets for s in starts]),
        quats=np.concatenate([s.quats for s in starts]),
        scenarios=[name for s in starts for name in s.scenarios],
    )


@dataclass
class DropResult:
    rows: list[dict]  # one per env
    steps: int
    sim_seconds: float  # wall time of `scene.step()` alone
    total_seconds: float  # including the metrics
    n_envs: int
    fps: float = field(init=False)  # simulated env-steps per second of `scene.step()`

    def __post_init__(self) -> None:
        self.fps = self.n_envs * self.steps / max(self.sim_seconds, 1e-9)

    def failures(self) -> list[str]:
        """Physics errors: penetration at rest, pegs still moving, a centred drop not ending in the hole."""
        errors = []
        for r in self.rows:
            where = f"env {r['env']} ({r['scenario']}, clearance {r['clearance_mm']:.2f} mm)"
            if r["rest_penetration_mm"] > PENETRATION_TOL * 1e3:
                errors.append(
                    f"{where}: penetration at rest {r['rest_penetration_mm']:.3f} mm"
                )
            if r["outcome"] == "moving":
                errors.append(f"{where}: still moving after the drop")
            if r["scenario"] == "centred" and r["outcome"] != "in_hole":
                errors.append(f"{where}: centred drop ended '{r['outcome']}'")
        return errors


class DropSim:
    """The board and one free peg, aimed at one slot, in each of `starts.n_envs` parallel environments."""

    def __init__(
        self, assets: PihAssets, cfg: DropCfg, starts: Starts, show_viewer: bool = False
    ):
        if not gs._initialized:
            raise RuntimeError("Call gs.init() before creating a DropSim")
        n_slots = len(assets.hole_frames)
        if starts.slots.min() < 0 or starts.slots.max() >= n_slots:
            raise ValueError(
                f"Start slots must lie in [0, {n_slots}), the board has {n_slots} slots"
            )
        self.assets, self.cfg, self.starts = assets, cfg, starts
        self.n_envs = starts.n_envs
        self.top = assets.cfg.board.thickness
        self.floor = self.top - assets.cfg.board.depth
        self.footprint = assets.geometry.board_footprint
        self.board_pose = (0.0, 0.0, self.top, 1.0, 0.0, 0.0, 0.0)
        self.slot_xy = np.array([f.pose[:2] for f in assets.hole_frames])[
            starts.slots
        ]  # (n_envs, 2)
        self.start_poses = self._start_poses()  # (n_envs, 7)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=cfg.ctrl_dt),
            # as in GenesisScene: Newton solver, the env's control step, then the config overrides
            rigid_options=gs.options.RigidOptions(
                dt=cfg.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                **(cfg.rigid_contact or {}),
            ),
            vis_options=gs.options.VisOptions(shadow=False),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.25, -0.35, 0.3),
                camera_lookat=(0.0, 0.0, 0.02),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.board = self._add(
            assets.board_urdf, self.board_pose, _BOARD_COLOR, fixed=True
        )
        self.peg = self._add(
            assets.peg_urdf, tuple(self.start_poses[0]), _PEG_COLOR, fixed=False
        )
        self.scene.build(n_envs=self.n_envs)
        self.reset()

    def _add(self, urdf, pose, color, fixed: bool) -> GenesisURDF:
        obj = GenesisURDF(
            gs_scene=self.scene,
            properties=ObjectProperties(
                dims=(), pose=pose, fixed=fixed, color=color, urdf_path=str(urdf)
            ),
            device=th.device("cpu"),
        )
        obj.create()
        return obj

    def _start_poses(self) -> np.ndarray:
        poses = np.zeros((self.n_envs, 7))
        peg_boxes = self.assets.geometry.peg_boxes
        for e in range(self.n_envs):
            quat = self.starts.quats[e]
            # lift so that the lowest corner of the (tilted) peg starts `hover` above the board
            lowest = world_vertices(peg_boxes, (0.0, 0.0, 0.0, *quat))[:, 2].min()
            xy = self.slot_xy[e] + self.starts.offsets[e]
            poses[e] = (*xy, self.top + self.cfg.hover - lowest, *quat)
        return poses

    def reset(self) -> None:
        # the whole scene first: solver state left over from a previous drop changes chaotic ones (a peg toppling)
        self.scene.reset()
        self.peg.set_pose(
            th.tensor(self.start_poses, dtype=gs.tc_float, device=gs.device)
        )
        self.peg._obj.zero_all_dofs_velocity()
        self.deepest = np.zeros(self.n_envs)
        self.last_moving = np.zeros(self.n_envs)  # [s] last time above REST_SPEED
        self.jitter = np.zeros(
            self.n_envs
        )  # [m/s] max speed within JITTER_WINDOW of the end
        self.t = 0.0
        self.steps = 0
        self.sim_seconds = 0.0

    def poses(self) -> np.ndarray:
        """(n_envs, 7) peg poses."""
        return self.peg.get_pose().cpu().numpy()

    def speeds(self) -> np.ndarray:
        """(n_envs,) linear speeds of the pegs."""
        return th.linalg.vector_norm(self.peg._obj.get_vel(), dim=-1).cpu().numpy()

    def _penetration(self, poses: np.ndarray) -> np.ndarray:
        geom = self.assets.geometry
        return max_penetration_batch(
            geom.peg_boxes, poses, geom.board_boxes, self.board_pose
        )

    def step(self) -> None:
        t0 = time.perf_counter()
        self.scene.step()
        self.sim_seconds += time.perf_counter() - t0
        self.steps += 1
        self.t += self.cfg.ctrl_dt

        self.deepest = np.maximum(self.deepest, self._penetration(self.poses()))
        speeds = self.speeds()
        self.last_moving[speeds > REST_SPEED] = self.t
        if self.t > self.cfg.settle - JITTER_WINDOW:
            self.jitter = np.maximum(self.jitter, speeds)

    def run(self) -> DropResult:
        self.reset()
        t0 = time.perf_counter()
        for _ in range(round(self.cfg.settle / self.cfg.ctrl_dt)):
            self.step()
        return self.result(total_seconds=time.perf_counter() - t0)

    def _outcome(self, e: int, pose: np.ndarray, lowest: float, tilt: float) -> str:
        fp = self.footprint
        if self.jitter[e] > REST_SPEED:
            return "moving"
        if (
            not (fp.x0 <= pose[0] <= fp.x1 and fp.y0 <= pose[1] <= fp.y1)
            and lowest < self.top - SEAT_TOL
        ):
            return "off_board"
        if abs(lowest - self.floor) < SEAT_TOL and tilt < UPRIGHT_TOL:
            return "in_hole"
        if lowest > self.top - SEAT_TOL:
            return "fallen" if tilt > FALLEN_TILT else "on_board"
        return "jammed"

    def result(self, total_seconds: float = 0.0) -> DropResult:
        poses = self.poses()
        rest = self._penetration(poses)
        clearances = self.assets.cfg.board.clearances
        peg_boxes, rows = self.assets.geometry.peg_boxes, []
        for e in range(self.n_envs):
            pose = poses[e]
            lowest = float(world_vertices(peg_boxes, pose)[:, 2].min())
            tilt = math.degrees(2 * math.acos(min(1.0, abs(float(pose[3])))))
            start_tilt = math.degrees(
                2 * math.acos(min(1.0, abs(float(self.starts.quats[e, 0]))))
            )
            rows.append(
                {
                    "env": e,
                    "slot": int(self.starts.slots[e]),
                    "scenario": self.starts.scenarios[e],
                    "clearance_mm": clearances[self.starts.slots[e]] * 1e3,
                    "start_dx_mm": self.starts.offsets[e, 0] * 1e3,
                    "start_dy_mm": self.starts.offsets[e, 1] * 1e3,
                    "start_tilt_deg": start_tilt,
                    "outcome": self._outcome(e, pose, lowest, tilt),
                    "transient_penetration_mm": self.deepest[e] * 1e3,
                    "rest_penetration_mm": rest[e] * 1e3,
                    "lowest_above_floor_mm": (lowest - self.floor) * 1e3,
                    "final_tilt_deg": tilt,
                    "final_xy_offset_mm": float(
                        np.linalg.norm(pose[:2] - self.slot_xy[e])
                    )
                    * 1e3,
                    "settle_time_s": self.last_moving[e],
                    "jitter_mm_s": self.jitter[e] * 1e3,
                }
            )
        return DropResult(
            rows=rows,
            steps=self.steps,
            sim_seconds=self.sim_seconds,
            total_seconds=total_seconds,
            n_envs=self.n_envs,
        )


def summarize(result: DropResult) -> list[dict]:
    """Per clearance: outcome counts, penetration and jitter over all envs."""
    out = []
    for clearance in sorted({r["clearance_mm"] for r in result.rows}, reverse=True):
        rows = [r for r in result.rows if r["clearance_mm"] == clearance]
        transient = np.array([r["transient_penetration_mm"] for r in rows])
        out.append(
            {
                "clearance_mm": clearance,
                "n": len(rows),
                **{o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES},
                "rest_penetration_max_mm": max(r["rest_penetration_mm"] for r in rows),
                "transient_penetration_max_mm": float(transient.max()),
                "transient_penetration_p95_mm": float(np.percentile(transient, 95)),
                "jitter_max_mm_s": max(r["jitter_mm_s"] for r in rows),
                "settle_time_max_s": max(r["settle_time_s"] for r in rows),
            }
        )
    return out
