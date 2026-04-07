# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "plotly",
#   "clearml",
# ]
# ///

import argparse
import logging
from typing import Optional

import matplotlib

from helpers.data_getter import DataGetter

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
matplotlib.use("Agg")  # non-interactive backend, safe in any env

# ---------------------------------------------------------------------------
# RUNNING SCRIPT IN CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    print(
        "▄▖▜       ▖  ▖▖   ▄▖      ▄▖▜   ▗ ▗     \n"
        "▌ ▐ █▌▀▌▛▘▛▖▞▌▌   ▙▖▚▘▛▌  ▙▌▐ ▛▌▜▘▜▘█▌▛▘\n"
        "▙▖▐▖▙▖█▌▌ ▌▝ ▌▙▖  ▙▖▞▖▙▌  ▌ ▐▖▙▌▐▖▐▖▙▖▌ \n"
        "                      ▌                 \n"
        "AGH Center of Excellence in Artificial Intelligence\n"
        "Maciej Aleksandrowicz 2026"
    )

    args = _build_parser().parse_args(argv)
    data = DataGetter(
        project_name=args.project_name,
        recursive_projects=True,
        max_samples=args.max_samples,
        metrics_paths=args.metric_paths,
        tags_select=["summary"] + args.tags,
    )
    if not data.tasks:
        print("No tasks)")
        return
    if not data.metrics_paths:
        print("No metrics")
        return

    print("TODO: implement reporting to the ClearML")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", help="ClearML project name.", required=True)
    p.add_argument(
        "--tags",
        nargs="+",
        help="One or more tags to select tasks (ClearML ANDs multiple tags).",
        required=False,
        default=[],
    )
    # p.add_argument("--exp-projects-names", help="One or more projects names (i.e. different experiments) which are already summarized.", required=True)
    p.add_argument(
        "--metrics",
        nargs="+",
        metavar="PATH",
        dest="metric_paths",
        help=(
            "Slash-separated metric paths to aggregate, e.g. "
            "'episode_reward/train/y'. "
            "Omit to auto-detect all metrics shared by every selected task."
        ),
    )
    p.add_argument(
        "--summary-task-name",
        default="EXPERIMENTS_SUMMARY",
        help="Name for the created summary task (default: SUMMARY).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=1_000,
        dest="max_samples",
        help="Maximum scalar samples fetched per task (default: 1000).",
    )
    p.add_argument(
        "--plots-backend",
        default="plotly",
        choices=["plotly", "matplotlib", "None"],
        help=("Backend to use for plotting (default: plotly)."),
    )
    p.add_argument("--cleanup-previous-tags", action="store_true", default=False)
    return p


if __name__ == "__main__":
    main()
