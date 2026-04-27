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


def build_parser(
    tags_required: bool, default_summary_task_name: str
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


def setup_logging() -> None:
    pass


def print_banner(text: str) -> None:
    pass


def common_parser_args(p: argparse.ArgumentParser) -> None:
    pass  # adds --metrics, --max-samples, --plots-backend, --cleanup-previous-tags


def timed(main_fn):
    pass  # decorator for the elapsed print
