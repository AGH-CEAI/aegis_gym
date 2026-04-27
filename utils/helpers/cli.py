# TODO implement for help with script
"""
> ## 2. Duplication between the two scripts
>
> `clearml_summarizer.py` and `clearml_exp_plotter.py` share:
> - ASCII banner print + author line
> - Logging setup (handler, formatter, basicConfig)
> - `matplotlib.use("Agg")`
> - Argument parser (~80% identical args)
> - `if __name__ == "__main__"` timing wrapper
>
> **Proposal:** extract a tiny `helpers/cli.py` with:
> ```python
> def setup_logging() -> None: ...
> def print_banner(text: str) -> None: ...
> def common_parser_args(p: argparse.ArgumentParser) -> None: ...  # adds --metrics, --max-samples, --plots-backend, --cleanup-previous-tags
> def timed(main_fn): ...  # decorator for the elapsed print
> ```
> Each script then keeps only its banner string and its specific args (`--tags` required vs optional, `--summary-task-name` default).
>
> This is the highest-leverage cleanup. ~50 lines saved, future changes happen in one place.
>
> ### 2a. Inconsistency to fix in passing
> - `--cleanup-previous-tags` has no `help=` in `clearml_summarizer.py`, has one in `clearml_exp_plotter.py`.
> - `--tags` is **required** in summarizer, **optional** in plotter. Intentional (plotter prepends `"summary"`)? Worth a comment.
>
> ---
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
