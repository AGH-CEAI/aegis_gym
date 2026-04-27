"""
> Be sure to get the ClearML project structure to this:
PROJECTS/.../
└── YOUR_PROJECT/
 ├── EXPERIMENT_1/
 │   ├── SUMMARY
 │   ├── trial_run_1
 │   ├── trial_run_2
 │   ├── ...
 │   └── trial_run_N
 ├── EXPERIMENT_2/
 │   ├── SUMMARY
 │   ├── trial_run_1
 │   ├── trial_run_2
 │   ├── ...
 │   └── trial_run_N
 ├── EXPERIMENT_3/
 │   ├── SUMMARY
 │   ├── trial_run_1
 │   ├── trial_run_2
 │   ├── ...
 │   └── trial_run_N
    └── EXPERIMENTS_SUMMARY
where `SUMMARY` is the result of the `clearml_summarizer.py` run in the `YOUR_PROJECT/EXPERIMENT_X` project name path, and the `EXPERIMENTS_SUMMARY` is the result of the `clearml_exp_plotter.py` run in the `YOUR_PROJECT` project name path.
"""

import argparse

from helpers.data_getter import SummaryType


def summary_type_parser(summary_type_strs: list[str]) -> list[SummaryType]:
    """Convert CLI args to SummaryType enum list"""
    types = []
    for s in summary_type_strs:
        # Handle comma-separated values in single arg
        for arg in s.split(","):
            arg = arg.strip().upper()
            try:
                types.append(SummaryType[arg])
            except KeyError:
                raise ValueError(
                    f"Invalid summary type: '{arg}'. "
                    f"Valid options: {', '.join(t.name for t in SummaryType)}"
                )
    return types


def build_parser(
    tags_required: bool,
    default_summary_task_name: str,
    default_summary_types: list[str],
) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project-name", help="ClearML project name.", required=True)
    p.add_argument(
        "--tags",
        nargs="+",
        help="One or more tags to select tasks (ClearML ANDs multiple tags).",
        required=tags_required,
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
        default=default_summary_task_name,
        help=f"Name for the created summary task (default: `{default_summary_task_name}`).",
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

    valid_types = ", ".join(t.name.lower() for t in SummaryType)
    p.add_argument(
        "--summary-types",
        nargs="+",
        type=summary_type_parser,
        default=default_summary_types,
        metavar="TYPE",
        help=(
            f"Summary statistics to compute. Valid: {valid_types}. "
            "Supports comma-separated (e.g., `--summary-types mean,mean_std`) "
            "or multiple args (e.g., `--summary-types mean --summary-types mean_std`). "
            f"Defaults to [`MEAN`] for a exp-summary and [`MEAN_STD`, `MEAN_MINMAX`] for a summary`."
        ),
    )
    return p
