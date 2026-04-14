#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
#   "joblib",
#   "matplotlib",
#   "numpy",
#   "plotly",
#   "tqdm",
# ]
# ///

import argparse
import logging
from time import perf_counter
from typing import Optional

import matplotlib

from helpers.data_getter import DataGetter, SummaryType
from helpers.summarizer import Summarizer
from helpers.logging_formatter import CustomFormatter

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(CustomFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
matplotlib.use("Agg")  # non-interactive backend, safe in any env


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
        max_samples=args.max_samples,
        recursive_projects=True,
        metrics_paths=args.metric_paths,
        tags_select=["summary"] + args.tags,
        merge_summaries_metrics=True,
    )
    if not data.tasks:
        print(">>> No tasks! Exiting.")
        return
    if not data.metrics_paths:
        print(">>> No metrics! Exiting.")
        return

    TAG_EXP_PLOTTER = "exp-summary"
    summarizer = Summarizer(
        tasks_data=data,
        summary_task_name=args.summary_task_name,
        plots_backend=args.plots_backend,
        summary_task_tags=[TAG_EXP_PLOTTER],
        experiments_summary_mode=True,
        summary_types=[SummaryType.MEAN],
    )
    summarizer.summarize(
        tag_for_tasks=TAG_EXP_PLOTTER,
        cleanup_previous_tags=args.cleanup_previous_tags,
        add_tag_to_tasks=True,
    )


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
        help="Name for the created summary task (default: `EXPERIMENTS_SUMMARY`).",
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
        help=("Backend to use for plotting (default: `plotly`)."),
    )
    p.add_argument(
        "--cleanup-previous-tags",
        action="store_true",
        default=False,
        help="Enable automatic remove of all `summary:XXX` tags from selected tasks.",
    )
    return p


if __name__ == "__main__":
    start = perf_counter()
    main()
    elapsed = perf_counter() - start
    print(f">>> Execution took {elapsed:.4f} s")
