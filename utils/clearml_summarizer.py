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

import logging
from time import perf_counter
from typing import Optional

import matplotlib

from helpers.cli import build_parser
from helpers.data_getter import DataGetter, NoMetricsError, NoTasksError
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
        "▄▖▜       ▖  ▖▖   ▄▖            ▘      \n"
        "▌ ▐ █▌▀▌▛▘▛▖▞▌▌   ▚ ▌▌▛▛▌▛▛▌▀▌▛▘▌▀▌█▌▛▘\n"
        "▙▖▐▖▙▖█▌▌ ▌▝ ▌▙▖  ▄▌▙▌▌▌▌▌▌▌█▌▌ ▌▙▖▙▖▌ \n"
        "AGH Center of Excellence in Artificial Intelligence\n"
        "Maciej Aleksandrowicz 2026"
    )

    args = build_parser(
        tags_required=True, default_summary_task_name="SUMMARY"
    ).parse_args(argv)
    try:
        data = DataGetter(
            project_name=args.project_name,
            max_samples=args.max_samples,
            metrics_paths=args.metric_paths,
            tags_select=args.tags,
        )
    except (NoTasksError, NoMetricsError):
        return

    summarizer = Summarizer(
        tasks_data=data,
        summary_task_name=args.summary_task_name,
        plots_backend=args.plots_backend,
    )
    summarizer.summarize(cleanup_previous_tags=args.cleanup_previous_tags)


if __name__ == "__main__":
    start = perf_counter()
    main()
    elapsed = perf_counter() - start
    print(f">>> Execution took {elapsed:.4f} s")
