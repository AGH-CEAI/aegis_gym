#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "clearml",
#   "joblib",
#   "matplotlib",
#   "numpy",
#   "plotly",
# ]
# ///
import logging
from typing import Optional

from helpers.cli import build_parser
from helpers.data_getter import DataGetter, NoMetricsError, NoTasksError, SummaryType
from helpers.summarizer import Summarizer
from helpers.logging import timed, setup_logging


@timed
def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    print(
        "#######################################################\n"
        "#       ▄▖▜       ▖  ▖▖   ▄▖            ▘             #\n"
        "#       ▌ ▐ █▌▀▌▛▘▛▖▞▌▌   ▚ ▌▌▛▛▌▛▛▌▀▌▛▘▌▀▌█▌▛▘       #\n"
        "#       ▙▖▐▖▙▖█▌▌ ▌▝ ▌▙▖  ▄▌▙▌▌▌▌▌▌▌█▌▌ ▌▙▖▙▖▌        #\n"
        "# AGH Center of Excellence in Artificial Intelligence #\n"
        "# ........... Maciej Aleksandrowicz 2026 ............ #\n"
        "#######################################################"
    )

    args = build_parser(
        tags_required=True,
        default_summary_task_name="SUMMARY",
        default_summary_types=[SummaryType.MEAN_MINMAX, SummaryType.MEAN_STD],
    ).parse_args(argv)
    try:
        data = DataGetter(
            project_name=args.project_name,
            max_samples=args.max_samples,
            metrics_paths=args.metric_paths,
            tags_select=args.tags,
        )
    except (NoTasksError, NoMetricsError):
        log = logging.getLogger(__name__)
        log.info("No tasks to process (check previous logs for more info). Exiting.")
        return

    summarizer = Summarizer(
        tasks_data=data,
        summary_task_name=args.summary_task_name,
        plots_backend=args.plots_backend,
        summary_types=args.summary_types,
    )
    summarizer.summarize(cleanup_previous_tags=args.cleanup_previous_tags)


if __name__ == "__main__":
    main()
