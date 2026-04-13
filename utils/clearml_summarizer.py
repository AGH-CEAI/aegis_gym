# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
#   "matplotlib",
#   "numpy",
#   "plotly",
#   "tqdm",
# ]
# ///

import argparse
import logging
from typing import Optional

import matplotlib

from helpers.data_getter import DataGetter
from helpers.summarizer import Summarizer

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
matplotlib.use("Agg")  # non-interactive backend, safe in any env


def main(argv: Optional[list[str]] = None) -> None:
    print(
        "▄▖▜       ▖  ▖▖   ▄▖            ▘      \n"
        "▌ ▐ █▌▀▌▛▘▛▖▞▌▌   ▚ ▌▌▛▛▌▛▛▌▀▌▛▘▌▀▌█▌▛▘\n"
        "▙▖▐▖▙▖█▌▌ ▌▝ ▌▙▖  ▄▌▙▌▌▌▌▌▌▌█▌▌ ▌▙▖▙▖▌ \n"
        "AGH Center of Excellence in Artificial Intelligence\n"
        "Maciej Aleksandrowicz 2026"
    )

    args = _build_parser().parse_args(argv)
    data = DataGetter(
        max_samples=args.max_samples,
        metrics_paths=args.metric_paths,
        project_name=args.project_name,
        tags_select=args.tags,
    )
    if not data.tasks:
        return
    if not data.metrics_paths:
        return
    summarizer = Summarizer(
        tasks_data=data,
        summary_task_name=args.summary_task_name,
        plots_backend=args.plots_backend,
    )
    summarizer.summarize(cleanup_previous_tags=args.cleanup_previous_tags)


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
        required=True,
    )
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
        default="SUMMARY",
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
