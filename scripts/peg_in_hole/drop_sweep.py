"""
Contact parameter sweep: runs `drop_test` for every combination of the grid in `drop_sweep.yaml` (or for the explicit
`configs` list, as in `drop_confirm.yaml`) and every seed, each in its own process (a new `ctrl_dt` needs a new
scene), and summarizes the runs per configuration and clearance.

Writes into `<out-dir>/sweep_<timestamp>/` (default `logs/pih_spike/`, ignored by git):
- `<tag>.csv` and `<tag>.log`: the per-env rows and the output of every run;
- `summary.csv` and `summary.md`: one row per configuration and clearance.

Run from the repo root, e.g.:
    python3 -m scripts.peg_in_hole.drop_sweep
    python3 -m scripts.peg_in_hole.drop_sweep --only dt4 --dry-run
    python3 -m scripts.peg_in_hole.drop_sweep --sweep scripts/peg_in_hole/drop_confirm.yaml
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

OUTCOMES = ("in_hole", "on_board", "fallen", "jammed", "off_board", "moving")
DEFAULT_SWEEP = Path(__file__).with_name("drop_sweep.yaml")
DEFAULT_OUT_DIR = Path("logs/pih_spike")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--sweep", type=Path, default=DEFAULT_SWEEP, help="grid definition (YAML)"
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--only",
        default=None,
        help="run only the configurations whose tag contains this text",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="list the configurations and exit"
    )
    return p.parse_args()


def _short(value) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return f"{value * 1e3:g}"  # seconds to ms
    return str(value)


_ABBREV = {
    "box_box_detection": "bbd",
    "constraint_timeconst": "tc",
    "noslip_iterations": "ns",
}


def _tag(ctrl_dt: float, contact: dict) -> str:
    return "_".join(
        [
            f"dt{_short(ctrl_dt)}",
            *(f"{_ABBREV.get(k, k)}{_short(v)}" for k, v in contact.items()),
        ]
    )


def configurations(sweep: dict) -> list[dict]:
    """
    The runs of a sweep: every combination of `grid`, or the explicit `configs` list, times every seed of
    `run.seed` (an int or a list). Each is {'tag', 'ctrl_dt', 'rigid_contact', 'seed'}.
    """
    base = []
    for c in sweep.get("configs", []):
        base.append((c.get("ctrl_dt", 0.004), dict(c.get("rigid_contact") or {})))
    if "grid" in sweep:
        grid = sweep["grid"]
        contact_grid = grid.get("rigid_contact", {})
        keys = list(contact_grid)
        for ctrl_dt in grid.get("ctrl_dt", [0.004]):
            for values in itertools.product(*(contact_grid[k] for k in keys)):
                base.append((ctrl_dt, dict(zip(keys, values, strict=True))))
    seeds = sweep.get("run", {}).get("seed", 0)
    seeds = seeds if isinstance(seeds, list) else [seeds]
    configs = []
    for ctrl_dt, contact in base:
        for seed in seeds:
            tag = _tag(ctrl_dt, contact) + (f"_s{seed}" if len(seeds) > 1 else "")
            configs.append(
                {"tag": tag, "ctrl_dt": ctrl_dt, "rigid_contact": contact, "seed": seed}
            )
    return configs


def run_config(config: dict, run: dict, out_dir: Path) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "scripts.peg_in_hole.drop_test",
        "--scenario",
        str(run.get("scenario", "random")),
        "-B",
        str(run.get("num_envs", 64)),
        "--seed",
        str(config["seed"]),
        "--settle",
        str(run.get("settle", 3.0)),
        "--ctrl-dt",
        str(config["ctrl_dt"]),
        "--rigid-contact",
        json.dumps(config["rigid_contact"]),
        "--tag",
        config["tag"],
        "--out-dir",
        str(out_dir),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        check=False,  # a failing configuration is a result, not an error of the sweep
        capture_output=True,
        text=True,
        env={**os.environ, "CLEARML_OFFLINE_MODE": "1"},
    )
    (out_dir / f"{config['tag']}.log").write_text(proc.stdout + proc.stderr)
    csvs = sorted(out_dir.glob(f"drop_{config['tag']}_*.csv"))
    return {
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - t0,
        "csv": csvs[-1] if csvs else None,
    }


def summarize(config: dict, outcome: dict) -> list[dict]:
    if outcome["csv"] is None:
        return [{"tag": config["tag"], "status": f"crashed ({outcome['returncode']})"}]
    with open(outcome["csv"]) as f:
        rows = list(csv.DictReader(f))
    out = []
    for clearance in sorted({float(r["clearance_mm"]) for r in rows}, reverse=True):
        rs = [r for r in rows if float(r["clearance_mm"]) == clearance]
        impact = sorted(float(r["transient_penetration_mm"]) for r in rs)
        out.append(
            {
                "tag": config["tag"],
                "status": "OK" if outcome["returncode"] == 0 else "FAILED",
                "ctrl_dt_ms": config["ctrl_dt"] * 1e3,
                "seed": config["seed"],
                **config["rigid_contact"],
                "clearance_mm": clearance,
                "n": len(rs),
                **{o: sum(r["outcome"] == o for r in rs) for o in OUTCOMES},
                "rest_pen_max_mm": round(
                    max(float(r["rest_penetration_mm"]) for r in rs), 3
                ),
                "impact_pen_max_mm": round(impact[-1], 3),
                "impact_pen_p95_mm": round(impact[int(0.95 * (len(impact) - 1))], 3),
                "jitter_max_mm_s": round(max(float(r["jitter_mm_s"]) for r in rs), 2),
                # pegs with a speed burst at rest: contact chatter, not a slow settling
                "jitter_over_50": sum(float(r["jitter_mm_s"]) > 50 for r in rs),
                "settle_max_s": round(max(float(r["settle_time_s"]) for r in rs), 2),
                "fps": int(float(rs[0]["fps"])),
                "run_s": round(outcome["seconds"], 1),
            }
        )
    return out


def write_summary(rows: list[dict], out_dir: Path) -> str:
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += [
        "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows
    ]
    table = "\n".join(lines)
    (out_dir / "summary.md").write_text(table + "\n")
    return table


def main() -> int:
    args = parse_args()
    with open(args.sweep) as f:
        sweep = yaml.safe_load(f)
    configs = [
        c for c in configurations(sweep) if args.only is None or args.only in c["tag"]
    ]
    print(f"{len(configs)} configurations from {args.sweep}")
    for c in configs:
        print(
            f"  {c['tag']}: ctrl_dt {c['ctrl_dt']}, rigid_contact {c['rigid_contact']}"
        )
    if args.dry_run:
        return 0

    out_dir = args.out_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, config in enumerate(configs, 1):
        outcome = run_config(config, sweep.get("run", {}), out_dir)
        rows += summarize(config, outcome)
        status = "OK" if outcome["returncode"] == 0 else f"exit {outcome['returncode']}"
        print(
            f"[{i}/{len(configs)}] {config['tag']}: {status}, {outcome['seconds']:.0f} s",
            flush=True,
        )
    print(write_summary(rows, out_dir))
    print(f"results: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
