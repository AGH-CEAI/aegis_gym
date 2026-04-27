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
from typing import Optional

import matplotlib

from helpers.cli import build_parser
from helpers.data_getter import DataGetter, SummaryType, NoTasksError, NoMetricsError
from helpers.summarizer import Summarizer
from helpers.logging import (
    CustomFormatter,
    timed,
    ignore_joblib_loky_semaphores_warnings,
)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(CustomFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
matplotlib.use("Agg")  # non-interactive backend, safe in any env


@timed
def main(argv: Optional[list[str]] = None) -> None:
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
        tags_required=False, default_summary_task_name="EXPERIMENTS_SUMMARY"
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


if __name__ == "__main__":
    ignore_joblib_loky_semaphores_warnings()
    main()
