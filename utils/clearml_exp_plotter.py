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
from helpers.data_getter import DataGetter, SummaryType, NoTasksError, NoMetricsError
from helpers.summarizer import Summarizer
from helpers.logging import timed, setup_logging


@timed
def main(argv: Optional[list[str]] = None) -> None:
    setup_logging()
    print(
        "#######################################################\n"
        "#      ▄▖▜       ▖  ▖▖   ▄▖      ▄▖▜   ▗ ▗            #\n"
        "#      ▌ ▐ █▌▀▌▛▘▛▖▞▌▌   ▙▖▚▘▛▌  ▙▌▐ ▛▌▜▘▜▘█▌▛▘       #\n"
        "#      ▙▖▐▖▙▖█▌▌ ▌▝ ▌▙▖  ▙▖▞▖▙▌  ▌ ▐▖▙▌▐▖▐▖▙▖▌        #\n"
        "#                            ▌                        #\n"
        "# AGH Center of Excellence in Artificial Intelligence #\n"
        "# ........... Maciej Aleksandrowicz 2026 ............ #\n"
        "#######################################################"
    )

    args = build_parser(
        tags_required=False,
        default_summary_task_name="EXPERIMENTS_SUMMARY",
        default_summary_types=[SummaryType.MEAN],
    ).parse_args(argv)
    try:
        data = DataGetter(
            project_name=args.project_name,
            max_samples=args.max_samples,
            recursive_projects=True,
            metrics_paths=args.metric_paths,
            tags_select=["summary"] + args.tags,
            merge_summaries_metrics=True,
        )
    except (NoTasksError, NoMetricsError):
        log = logging.getLogger(__name__)
        log.info("No tasks to process (check previous logs for more info). Exiting.")
        return

    TAG_EXP_PLOTTER = "exp-summary"
    summarizer = Summarizer(
        tasks_data=data,
        summary_task_name=args.summary_task_name,
        plots_backend=args.plots_backend,
        summary_task_tags=[TAG_EXP_PLOTTER],
        experiments_summary_mode=True,
        summary_types=args.summary_types,
    )
    summarizer.summarize(
        tag_for_tasks=TAG_EXP_PLOTTER,
        cleanup_previous_tags=args.cleanup_previous_tags,
        add_tag_to_tasks=True,
    )


if __name__ == "__main__":
    main()
