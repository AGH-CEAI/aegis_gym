#!/usr/bin/env python3
"""
Summarize ClearML tasks (selected by tags) by aggregating scalar metrics.

Aggregation produces mean, std, min, and max curves across all selected tasks,
and logs them back to ClearML as a dedicated summary task.

Metric paths follow ClearML's nested scalar dict structure:
    scalars[title][series]["x" | "y"]

Usage examples
--------------
# Explicit metrics:
python clearml_summarize.py MyProject/Path tag1 tag2 \
    --metrics "episode_reward/train/y" "loss/critic/y"

# Auto-detect all metrics shared by every selected task:
python clearml_summarize.py MyProject/Path tag1 tag2 \
    --summary-task-name "experiment_SUMMARY"
"""

import argparse
import logging
from typing import Iterable, Optional

import numpy as np
from clearml import Task, TaskTypes

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
log = logging.getLogger(__name__)


# TODO use metric_paths dict to create aggregation with name of the key of the data under the metric path.
# If the metric paths are not given, automatically deduce all available metrics shared between the tasks and aggregate them.
def summarize(
    project_name: str,
    tags_select: Iterable[str],
    metric_paths: Optional[list[str]] = None,
    max_samples: int = 1_000,
    summary_task_name: str = "SUMMARY",
) -> None:
    """Aggregate scalar metrics across ClearML tasks and log a summary task.

    Args:
        project_name:       ClearML project to create the summary task in.
        tags_select:        Tags used to query tasks (ClearML ANDs multiple tags).
        metric_paths:       List of slash-separated metric paths, e.g.
                            ``["episode_reward/train/y", "loss/critic/y"]``.
                            If *None*, all paths shared by every selected task
                            are detected automatically.
        max_samples:        Maximum scalar samples fetched per task.
                            A warning is emitted when a task likely exceeds this.
        summary_task_name:  Name for the resulting summary task.
    """
    tags_select = list(set(tags_select))
    tag_filter = _build_tag_filter(tags_select)
    log.info(
        f"Querying tasks with tags {tags_select} (AND) in project '{project_name}'"
    )

    tasks = Task.get_tasks(project_name=project_name, tags=tag_filter)
    if not tasks:
        log.warning(
            f"No tasks found for tags {tags_select} (AND) - nothing to summarise"
        )
        return
    log.info(f"Found {len(tasks)} task(s)")

    scalars_per_task: list[dict] = [None] * len(tasks)
    for cnt, t in enumerate(tasks):
        data = t.get_reported_scalars(max_samples=max_samples)
        _warn_if_truncated(task=t, scalars=data, max_samples=max_samples)

        for k in list(data.keys()):
            if k.startswith(":"):
                data.pop(k)

        scalars_per_task[cnt] = data

    if metric_paths is None:
        metric_paths = _detect_common_metric_paths(tasks, scalars_per_task)
        if not metric_paths:
            log.error("No scalar metrics shared by all selected tasks. Aborting")
            return
        log.info(f"Auto-detected {len(metric_paths)} shared metric path(s):")
        for p in metric_paths:
            log.info(f"\t{p}")
        log.info(f"Reduced number of processing tasks to: {len(tasks)}")
    else:
        _validate_metric_paths(metric_paths, scalars_per_task)

    summary_task = Task.init(
        project_name=project_name,
        task_name=summary_task_name,
        task_type=TaskTypes.application,
        tags=["summary"],
        reuse_last_task_id=False,
        auto_resource_monitoring=False,
    )
    summary_logger = summary_task.get_logger()

    # TODO add flag to optionally turn it off
    for t in tasks:
        t.add_tags([f"summary:{summary_task.task_id}"])

    failed_cnt = 0
    try:
        for path_str in metric_paths:
            path = path_str.split("/")
            log.info(f"Aggregating metric: {path_str}")

            curves_y, curves_x = _extract_curves(scalars_per_task, path)
            if curves_y is None:
                log.warning(
                    f"Skipping '{path_str}': extraction failed for one or more tasks"
                )
                failed_cnt += 1
                continue

            y_array = _pad_to_array(curves_y)
            x_axis = curves_x[
                int(np.argmax([len(x) for x in curves_x]))
            ]  # the longest x

            mean_y = np.mean(y_array, axis=0)
            std_y = np.std(y_array, axis=0)
            min_y = np.min(y_array, axis=0)
            max_y = np.max(y_array, axis=0)

            # Use a clean prefix derived from the path for ClearML titles
            # e.g. "episode_reward/train/y" → title "episode_reward", series "train"
            agg_title, agg_series = _path_to_title_series(path_str)

            for step, (s_mean, s_std, s_min, s_max) in enumerate(
                zip(mean_y, std_y, min_y, max_y)
            ):
                x = x_axis[step] if step < len(x_axis) else step
                summary_logger.report_scalar(
                    f"{agg_title}/{agg_series}_mean-min-max", "mean", s_mean, x
                )
                summary_logger.report_scalar(
                    f"{agg_title}/{agg_series}_mean-min-max", "min", s_min, x
                )
                summary_logger.report_scalar(
                    f"{agg_title}/{agg_series}_mean-min-max", "max", s_max, x
                )
                summary_logger.report_scalar(
                    f"{agg_title}/{agg_series}_mean-std", "mean", s_mean, x
                )
                summary_logger.report_scalar(
                    f"{agg_title}/{agg_series}_mean-std", "std+", s_mean + s_std, x
                )
                summary_logger.report_scalar(
                    f"{agg_title}/{agg_series}_mean-std", "std-", s_mean - s_std, x
                )

        # Persist configuration for reproducibility
        summary_task.set_parameter("summarize/tags_select", str(tags_select))
        summary_task.set_parameter("summarize/metric_paths", str(metric_paths))
        summary_task.set_parameter("summarize/max_samples", max_samples)
        summary_task.set_parameter("summarize/n_source_tasks", len(tasks))

        log.info(
            f"Summary task '{project_name}/{summary_task_name}' (id={summary_task.task_id}) created from {len(tasks)} source task(s)"
        )
        if failed_cnt:
            log.warning(
                f"Summarization failed for {failed_cnt} out of {len(metric_paths)} metric(s)"
            )

    except Exception:
        log.exception("Summarisation failed - marking summary task as failed")
        summary_task.mark_failed(force=True)
        raise
    finally:
        summary_task.close()


def _build_tag_filter(tags: Iterable[str]) -> list[str]:
    """Return a ClearML tag filter that requires ALL tags to be present (AND).

    ClearML's default behaviour is OR. Prepending ``"__$all"`` switches the
    default operator to AND for every tag that follows.

    See: https://clear.ml/docs/latest/docs/clearml_sdk/task_sdk/#tag-filters
    """
    tags = list(tags)
    if not tags:
        raise ValueError("At least one tag must be provided.")
    if len(tags) == 1:
        return tags  # AND/OR is irrelevant for a single tag
    return ["__$all"] + tags


def _warn_if_truncated(task: Task, scalars: dict, max_samples: int) -> None:
    """Emit a warning if any series in *scalars* appears to be at the sample cap."""
    for title, series_dict in scalars.items():
        for series, data in series_dict.items():
            y = data.get("y", [])
            if len(y) >= max_samples:
                log.warning(
                    f"Task {task.id} metric '{title}/{series}' has {len(y)} samples - may be truncated "
                    f"(max_samples={max_samples}). Consider increasing --max-samples",
                )


def _filter_scalars(scalars: dict) -> dict:
    """Remove scalar titles that begin with ':'.

    ClearML uses ':'-prefixed titles for internal/system metrics
    (e.g. ':monitor:gpu', ':monitor:cpu') that are not user-reported
    and should be excluded from aggregation.
    """
    return {
        title: series_dict
        for title, series_dict in scalars.items()
        if not title.startswith(":")
    }


def _detect_common_metric_paths(
    tasks: list[Task],
    scalars_per_task: list[dict],
) -> list[str]:
    """Return slash-separated metric paths present in *every* task's scalars.

    ClearML scalar structure: scalars[title][series] = {"x": [...], "y": [...]}
    We enumerate all (title, series) pairs and keep those shared by all tasks.
    The leaf key is always "y" (values); "x" is the corresponding step axis.
    """
    sets: list[set[str]] = []

    for cnt, (task, scalars) in enumerate(zip(tasks, scalars_per_task)):
        paths: set[str] = set()
        for title, series_dict in scalars.items():
            for series in series_dict:
                paths.add(f"{title}/{series}/y")

        if not paths:
            log.warning(f"Task {task.id} reported no scalar metrics — skipping.")
            tasks.pop(cnt)
            scalars_per_task.pop(cnt)
            continue

        sets.append(paths)

    common = sets[0].intersection(*sets[1:])
    return sorted(common)


def _validate_metric_paths(
    metric_paths: list[str],
    scalars_per_task: list[dict],
) -> None:
    """Log a warning for any explicitly requested path missing in some task."""
    for path_str in metric_paths:
        path = path_str.split("/")
        for idx, scalars in enumerate(scalars_per_task):
            try:
                _access_path(scalars, path)
            except KeyError:
                log.warning(
                    f"Metric path '{path_str}' not found in task index {idx} — it will be skipped",
                )


def _access_path(data: dict, path: list[str]):
    """Traverse a nested dict by a list of keys, raising KeyError if missing."""
    val = data
    for key in path:
        val = val[key]
    return val


def _extract_curves(
    scalars_per_task: list[dict],
    path: list[str],
) -> tuple[Optional[list[list[float]]], Optional[list[list[float]]]]:
    """Extract y-value and x-axis curves from every task for the given path.

    The path should end in "y"; the x-axis is read from the sibling "x" key.
    Returns (None, None) if any task is missing the path.
    """
    curves_y: list[list[float]] = []
    curves_x: list[list[float]] = []

    x_path = path[:-1] + ["x"]

    # print(scalars_per_task)
    for cnt, scalars in enumerate(scalars_per_task):
        # print(f"CNT: {cnt} | scalars: {scalars}")
        if not scalars:
            continue
        try:
            y = list(_access_path(scalars, path))
        except KeyError:
            print(f"CNT: {cnt} | Failed to access scalar y: {path}")
            return None, None
        try:
            x = list(_access_path(scalars, x_path))
        except KeyError:
            print(f"CNT: {cnt} | Failed to access scalar x: {x_path}")
            return None, None
        curves_y.append(y)
        curves_x.append(x)

    return curves_y, curves_x


def _pad_to_array(curves: list[list[float]]) -> np.ndarray:
    """Pad shorter curves with their last value and stack into a 2-D array."""
    max_len = max(len(c) for c in curves)
    padded = [None] * len(curves)
    for i, c in enumerate(curves):
        arr = np.asarray(c, dtype=float)
        if len(arr) < max_len:
            pad_value = arr[-1] if len(arr) > 0 else 0.0
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=pad_value)
        padded[i] = arr
    return np.array(padded)


def _path_to_title_series(path_str: str) -> tuple[str, str]:
    """Convert a slash path to a (ClearML title, series) pair for the summary.

    ``"episode_reward/train/y"``  ->  ``("episode_reward", "train")``
    ``"loss/y"``                  ->  ``("loss", "default")``
    """
    parts = path_str.rstrip("/y").split("/")
    if len(parts) >= 2:
        return "/".join(parts[:-1]), parts[-1]
    return parts[0], "default"


# ---------------------


def access_metric_path(data: dict, path: Iterable[str]) -> any:
    val = data
    for key in path:
        val = val[key]
    return val


def curves_to_arrays(all_curves: list) -> np.ndarray:
    if all_curves is None or None in all_curves:
        print("Failed to convert data into np.array (None detected).")
        return np.array([])

    max_len = max([len(c) for c in all_curves])
    padded_curves = [None] * len(all_curves)
    for cnt, c in enumerate(all_curves):
        padded = c
        if len(c) < max_len:
            pad_value = c[-1] if c else 0
            padded = np.pad(c, (0, max_len - len(c)), constant_values=pad_value)

        padded_curves[cnt] = padded
    return np.array(padded_curves)


# ---------------------------------------------------------------------------
# RUNNING SCRIPT IN CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    summarize(
        project_name=args.project_name,
        tags_select=args.tags,
        metric_paths=args.metric_paths,
        max_samples=args.max_samples,
        summary_task_name=args.summary_task_name,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("project_name", help="ClearML project name.")
    p.add_argument(
        "tags",
        nargs="+",
        help="One or more tags to select tasks (ClearML ANDs multiple tags).",
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
    return p


if __name__ == "__main__":
    main()
