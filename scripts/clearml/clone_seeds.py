"""
Clones a ClearML template task once per seed, for runs that differ by the seed only (publishable
seed statistics). The ClearML agent ignores the `Args` overrides (the stored configuration is
read-only there, see `ConfigManager._find_ignored_launch_args()`), so the seed is written into the
`rl` configuration of each clone (`rl.seed`, used by `gs.init()` and the runners); `Args/seed` is
set too, only to show it in the UI.

The clones are drafts unless `--queue` is given. Each one is read back to check its stored seed.

Run from the repo root:
    python -m scripts.clearml.clone_seeds --template <task id> --seeds 0-19 \
        --name PUSH_T_ADMIT_02_PPO_rl --project T-PUSHER_ADMITTANCE/007_RL_vel02ms --queue all
"""

import argparse
import re

from clearml import Task
from clearml.backend_api.session.client import APIClient

SEED_LINE = re.compile(r"(?m)^seed\s*=\s*.*$")


def parse_seeds(text: str) -> list[int]:
    """`0-4,10,12` -> [0, 1, 2, 3, 4, 10, 12]"""
    seeds = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        seeds += range(int(first), int(last or first) + 1)
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Repeated seeds in {text!r}.")
    return seeds


def with_seed(rl_config: str, seed: int) -> str:
    new, num = SEED_LINE.subn(f"seed = {seed}", rl_config)
    if num != 1:
        raise ValueError(
            f"Expected one top-level `seed` in the `rl` configuration, found {num}."
        )
    return new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--template", required=True, help="the task id to clone")
    p.add_argument("--seeds", required=True, help="e.g. 0-19 or 0,3,7")
    p.add_argument("--name", required=True, help="the clones are named <name>_<seed>")
    p.add_argument("--project", default=None, help="default: the template's project")
    p.add_argument("--queue", default=None, help="enqueue the clones (default: drafts)")
    args = p.parse_args()

    template = Task.get_task(task_id=args.template)
    rl_config = template.get_configuration_object("rl")
    if rl_config is None:
        raise ValueError(f"The template {args.template} has no `rl` configuration.")
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        with_seed(rl_config, seed)  # fails before any clone is made

    project = template.project
    if args.project:
        project = (
            Task.get_project_id(args.project)
            or APIClient().projects.create(name=args.project).id
        )
    for seed in seeds:
        clone = Task.clone(
            source_task=template, name=f"{args.name}_{seed}", project=project
        )
        clone.set_configuration_object(
            name="rl", config_text=with_seed(rl_config, seed)
        )
        clone.set_parameter("Args/seed", str(seed))
        stored = Task.get_task(task_id=clone.id).get_configuration_object_as_dict("rl")[
            "seed"
        ]
        if stored != seed:
            raise RuntimeError(f"{clone.id}: stored rl.seed {stored}, expected {seed}.")
        if args.queue:
            Task.enqueue(clone, queue_name=args.queue)
        print(
            f"{clone.name} {clone.id} rl.seed={stored}"
            + (f" -> {args.queue}" if args.queue else "")
        )


if __name__ == "__main__":
    main()
