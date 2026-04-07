# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "matplotlib",
#   "plotly",
#   "clearml",
# ]
# ///

import copy
import argparse
import logging
from typing import Iterable, Optional, Literal

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from clearml import Task, TaskTypes, Logger

logging.basicConfig(
    level=logging.INFO,
    # format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    format="%(asctime)s [%(levelname)s]: %(message)s",
)
matplotlib.use("Agg")  # non-interactive backend, safe in any env


class DataGetter:
    def __init__(
        self,
        project_name: str,
        tags_select: Iterable[str],
        metrics_paths: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ):
        self.log = logging.getLogger(__name__)
        self.project_name = project_name
        self.max_samples = max_samples

        self.tags_raw = tags_select
        self.tags_filter = self._prepare_tags_filter(tags_select)
        self.tasks: dict[str, dict] = self._create_tasks_data_dict()
        if metrics_paths is None:
            metrics_paths = self._get_common_metrics()
        else:
            self._select_metrics(metrics_paths)
        self.metrics_paths = metrics_paths

        if not self.tasks:
            self.log.error("Failed to found valid tasks.")
            return
        if not self.metrics_paths:
            self.log.error("Failed to select any valid metrics.")
            return
        self.log.info(
            f"Prepared {len(self.tasks)} tasks for summarization with {len(self.metrics_paths)} metrics."
        )

    def _prepare_tags_filter(self, tags: Iterable[str]) -> list[str]:
        tags = list(set(tags))
        if not tags:
            raise ValueError("At least one tag must be provided.")
        # Switch default ClearML behaviour from OR to AND with "_$all"
        return ["__$all"] + tags

    def _create_tasks_data_dict(
        self,
        only_completed: bool = True,
        remove_colon_scalars: bool = True,
        same_n_iterations: bool = True,
    ) -> dict[str, dict]:
        # Returns: dict[task_id, task_data_dict]

        tasks = {}
        tasks_raw = self._get_cleraml_tasks(self.project_name, self.tags_filter)
        if tasks_raw is None:
            return tasks

        self.min_n_iterations = tasks_raw[0].get_last_iteration()

        for t in tasks_raw:
            try:
                if only_completed and t.get_status() != "completed":
                    raise ValueError("Task is not completed.")

                data = self._task_data_getter(t)
                if remove_colon_scalars:
                    data = self._remove_colcon_scalars(data)

                if same_n_iterations:
                    self._compare_n_iterations(t.get_last_iteration())

                tasks[t.task_id] = data
            except ValueError as e:
                self.log.warning(f"Skipping task id {t.task_id} due to: {str(e)}")
        return tasks

    def _get_cleraml_tasks(self, project_name: str, tags: list[str]) -> list[Task]:
        self.log.info(
            f"Querying tasks with tags {self.tags_raw} (AND) in project '{self.project_name}'."
        )
        tasks = Task.get_tasks(project_name=project_name, tags=self.tags_filter)
        if tasks:
            self.log.info(f"Found {len(tasks)} task(s).")
            return tasks

        self.log.warning(
            f"No tasks found for tags {self.tags_filter} (AND) - nothing to summarise."
        )
        return None

    def _task_data_getter(self, t: Task) -> dict:
        if self.max_samples is None:
            return t.get_all_reported_scalars()
        return t.get_reported_scalars(max_samples=self.max_samples)

    def _remove_colcon_scalars(self, data: dict) -> dict:
        for k in list(data.keys()):
            if k.startswith(":"):
                data.pop(k)
        return data

    def _compare_n_iterations(self, n_iterations: int) -> None:
        if self.min_n_iterations < n_iterations:
            raise ValueError("Too few data samples.")

        if self.min_n_iterations > n_iterations:
            self.min_n_iterations = n_iterations

    def _get_common_metrics(self) -> list[str]:
        self.log.info("Auto-detecteding shared metric path(s).")
        metric_paths = self._detect_common_metric_paths()
        if not metric_paths:
            self.log.error(
                "No scalar metrics are shared by all selected tasks. Returning empty list."
            )
            return []

        self.log.info(f"Auto-detected {len(metric_paths)} shared metric path(s):")
        for p in metric_paths:
            self.log.info(f"\t{p}")

        return metric_paths

    def _detect_common_metric_paths(
        self,
        skip_time_series: bool = True,
    ) -> list[str]:
        # We need a shallow copy to modify the original dictionary
        tasks = copy.copy(self.tasks)
        sets: list[set[str]] = []

        for t_id, t_data in tasks.items():
            if not t_data:
                self.log.warning(
                    f"Task {t_id} reported no scalar metrics. Omitting it from analysis."
                )
                self.tasks.pop(t_id)
                continue

            paths: set[str] = set()
            for title, series_dict in t_data.items():
                if skip_time_series and "time" in title.lower():
                    continue
                for series in series_dict:
                    paths.add(f"{title}/{series}/y")

            sets.append(paths)

        if not sets:
            self.log.error("The given list of tasks do not have any metrics.")
            return []

        common_paths = sets[0].intersection(*sets[1:])
        return sorted(common_paths)

    def _select_metrics(self, metric_paths: list[str]):
        for path_str in metric_paths:
            path = path_str.split("/")

            # We need a CURRENT shallow copy to modify the original dictionary
            tasks = copy.copy(self.tasks)
            for t_id, t_data in tasks.items():
                if not t_data:
                    self.log.warning(
                        f"Task {t_id} reported no scalar metrics. Omitting it from analysis."
                    )
                    self.tasks.pop(t_id)
                    continue

                try:
                    val = t_data
                    for key in path:
                        val = val[key]
                except KeyError:
                    self.log.warning(
                        f"Metric path '{path_str}' not found in the task id {t_id}. This task will be omitted.",
                    )
                    self.tasks.pop(t_id)


class Summarizer:
    def __init__(
        self,
        tasks_data: DataGetter,
        summary_task_name: str = "SUMMARY",
        plots_backend: Literal["matplotlib", "plotly", "None"] = "plotly",
    ):
        """
        plots_backend:
            "matplotlib" – raster PNG stored in DEBUG SAMPLES tab
            "plotly"     – interactive vector figure stored in PLOTS tab
        """

        self.log = logging.getLogger(__name__)
        self.plots_backend = plots_backend

        self.tasks = tasks_data.tasks
        self.metric_paths = tasks_data.metrics_paths
        self.project_name = tasks_data.project_name
        self.i_tags_filter = tasks_data.tags_filter
        self.i_max_samples = tasks_data.max_samples

        self.summary_task_name = summary_task_name
        self.summary_task_type = TaskTypes.application
        self.summary_task_tags = ["summary"]

        self.plot_alpha_fill = 0.25
        self.plot_color_mean = "#2271B5"
        self.plot_color_band = "#6BAED6"
        self.plot_fig_size = (9, 4)

        self.log.info(f"Using `{self.plots_backend}` for plots backend.")

    def summarize(
        self,
        add_tag_to_tasks: bool = True,
        cleanup_previous_tags: bool = True,
    ) -> None:
        summary_task = Task.init(
            project_name=self.project_name,
            task_name=self.summary_task_name,
            task_type=self.summary_task_type,
            tags=self.summary_task_tags,
            reuse_last_task_id=False,
            auto_resource_monitoring=False,
        )

        self.log.info(
            f"Summary task '{self.project_name}/{self.summary_task_name}' (id={summary_task.task_id}) created."
        )

        if cleanup_previous_tags:
            self._cleanup_previous_tags()

        if add_tag_to_tasks:
            tags = [f"summary:{summary_task.task_id}"]
            for t_id in self.tasks.keys():
                t = Task.get_task(task_id=t_id)
                t.add_tags(tags)
            self.log.info(f"Added tag(s) {tags} to {len(self.tasks)} source tasks.")

        try:
            self._summarize(summary_task)
        except Exception as e:
            self.log.exception(f"Caught an exception: {e}")
        finally:
            summary_task.close()

        self.log.info(
            f"Finished summarization of {len(self.tasks)} tasks with {len(self.metric_paths)} metrics."
        )

    def _cleanup_previous_tags(self) -> None:
        self.log.info("Removing previous summary tags from task(s).")

        cleaned_tasks = 0
        removed_total = 0

        for t_id in self.tasks.keys():
            t = Task.get_task(task_id=t_id)

            current_tags = list(t.get_tags() or [])
            filtered_tags = [
                tag for tag in current_tags if not tag.startswith("summary:")
            ]

            if filtered_tags != current_tags:
                removed = [tag for tag in current_tags if tag.startswith("summary:")]
                t.set_tags(filtered_tags)
                cleaned_tasks += 1
                removed_total += len(removed)
                self.log.debug(f"Removed summary tags from task id {t_id}: {removed}")

        self.log.info(
            f"Removed {removed_total} summary tag(s) from {cleaned_tasks} task(s)."
        )

    def _summarize(self, summary_task: Task) -> None:
        something_failed = False

        # Persist configuration for reproducibility
        summary_task.set_parameter("summarize/tags_filter", str(self.i_tags_filter))
        summary_task.set_parameter("summarize/metric_paths", str(self.metric_paths))
        summary_task.set_parameter("summarize/max_samples", self.i_max_samples)
        summary_task.set_parameter("summarize/n_source_tasks", len(self.tasks))

        for cnt, path_str in enumerate(self.metric_paths):
            self.log.info(
                f"Aggregating metric {cnt + 1}/{len(self.metric_paths)}: {path_str}"
            )
            path_parts = path_str.split("/")

            series_y, axis_x = self._extract_series(path_parts)
            if series_y is None:
                self.log.warning(f"Skipping '{path_str}': extraction failed.")
                self.log.debug(
                    f"The content of series_y (len: {len(series_y)}): {series_y}"
                )
                something_failed = True
                continue

            summary = {}
            summary["mean_y"] = np.mean(series_y, axis=0)
            summary["std_y"] = np.std(series_y, axis=0)
            summary["min_y"] = np.min(series_y, axis=0)
            summary["max_y"] = np.max(series_y, axis=0)

            summary_logger = summary_task.get_logger()
            self._report_scalars(summary_logger, axis_x, summary, path_str)
            self._report_filled_plots(summary_logger, axis_x, summary, path_str)

        if something_failed:
            self.log.warning(
                ">>>>>>>>>> SOMETHING WENT WRONG. CHECK THE LOGS! <<<<<<<<<<"
            )

    def _extract_series(
        self,
        path_parts: list[str],
    ) -> tuple[list[list[float]], list[float]]:

        top_key = "/".join(path_parts[:-2])  # "Loss/action"

        series_y: list[list[float]] = []
        axis_x: list[float] = None

        for t_data in self.tasks.values():
            metric = t_data[top_key]["series"]
            series_y.append(list(metric["y"]))

            if axis_x is None:
                axis_x = list(metric["x"])

        return series_y, axis_x

    @staticmethod
    def _path_to_title_series(path_str: str) -> tuple[str, str]:
        # Use a clean prefix derived from the path for ClearML titles
        # e.g. "episode_reward/train/y" → title "episode_reward", series "train"
        parts = path_str.rstrip("/y").split("/")
        if len(parts) >= 2:
            return "/".join(parts[:-1]), parts[-1]
        return parts[0], "default"

    def _report_scalars(
        self,
        t_log: Logger,
        axis_x: list,
        summary: dict,
        path_str: str,
    ) -> None:
        """
        Upload summary to a ClearML task via its logger.
        """
        title, _ = self._path_to_title_series(path_str)

        mean_y = summary["mean_y"]
        std_y = summary["std_y"]
        min_y = summary["min_y"]
        max_y = summary["max_y"]

        for step in range(len(axis_x)):
            x = axis_x[step]

            t_log.report_scalar(f"{title}_mean-min-max", "mean", mean_y[step], x)
            t_log.report_scalar(f"{title}_mean-min-max", "min", min_y[step], x)
            t_log.report_scalar(f"{title}_mean-min-max", "max", max_y[step], x)
            t_log.report_scalar(f"{title}_mean-std", "mean", mean_y[step], x)
            t_log.report_scalar(
                f"{title}_mean-std", "std+", mean_y[step] + std_y[step], x
            )
            t_log.report_scalar(
                f"{title}_mean-std", "std-", mean_y[step] - std_y[step], x
            )

    def _report_filled_plots(
        self,
        t_log: Logger,
        axis_x: list,
        summary: dict,
        path_str: str,
    ) -> None:
        """
        Create filled confidence-band figures and upload them to a ClearML task.
        """
        x = np.asarray(axis_x)
        mean_y = np.asarray(summary["mean_y"])
        std_y = np.asarray(summary["std_y"])
        min_y = np.asarray(summary["min_y"])
        max_y = np.asarray(summary["max_y"])

        match self.plots_backend:
            case "plotly":
                self._report_filled_plots_plotly(
                    t_log,
                    path_str,
                    x,
                    mean_y,
                    std_y,
                    min_y,
                    max_y,
                )
            case "matplotlib":
                self._report_filled_plots_matplotlib(
                    t_log,
                    path_str,
                    x,
                    mean_y,
                    std_y,
                    min_y,
                    max_y,
                )
            case "None":
                pass
            case _:
                self.log.warning(
                    f"Unregonized plots backend `{self.plots_backend}` (Available: `plotly`,`matplotlib`). Skipping plots."
                )

    def _report_filled_plots_plotly(
        self,
        t_log: Logger,
        path_str: str,
        x,
        mean_y,
        std_y,
        min_y,
        max_y,
    ) -> None:
        agg_title, agg_series = self._path_to_title_series(path_str)

        def _hex_to_rgba(hex_color: str, alpha: float) -> str:
            hex_color = hex_color.lstrip("#")
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            return f"rgba({r},{g},{b},{alpha})"

        band_rgba = _hex_to_rgba(self.plot_color_band, self.plot_alpha_fill)

        # Plot A: mean ± std
        fig_std = go.Figure()

        fig_std.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean_y + std_y, (mean_y - std_y)[::-1]]),
                fill="toself",
                fillcolor=band_rgba,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="mean ± std",
            )
        )
        fig_std.add_trace(
            go.Scatter(
                x=x,
                y=mean_y,
                line=dict(color=self.plot_color_mean, width=2),
                hovertemplate="mean: %{y:.4f}<extra></extra>",
                name="mean",
            )
        )
        fig_std.update_layout(
            title=f"{agg_title} – {agg_series}  [mean ± std]",
            xaxis_title="step",
            yaxis_title="value",
        )

        t_log.report_plotly(
            title=f"{agg_title}_mean-std",
            series=f"{agg_title}_mean-std",
            figure=fig_std,
        )

        # Plot B: mean / min / max
        fig_mm = go.Figure()

        fig_mm.add_trace(
            go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([max_y, min_y[::-1]]),
                fill="toself",
                fillcolor=band_rgba,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="min – max range",
            )
        )
        fig_mm.add_trace(
            go.Scatter(
                x=x,
                y=mean_y,
                line=dict(color=self.plot_color_mean, width=2),
                hovertemplate="mean: %{y:.4f}<extra></extra>",
                name="mean",
            )
        )
        fig_mm.update_layout(
            title=f"{agg_title} – {agg_series}  [mean / min / max]",
            xaxis_title="step",
            yaxis_title="value",
        )

        t_log.report_plotly(
            title=f"{agg_title}_mean-min-max",
            series=f"{agg_title}_mean-min-max",
            figure=fig_mm,
        )

    def _report_filled_plots_matplotlib(
        self,
        t_log: Logger,
        path_str: str,
        x,
        mean_y,
        std_y,
        min_y,
        max_y,
    ) -> None:
        agg_title, agg_series = self._path_to_title_series(path_str)

        # Plot A: mean ± std
        fig_std, ax_std = plt.subplots(figsize=self.plot_fig_size)
        ax_std.fill_between(
            x,
            mean_y - std_y,
            mean_y + std_y,
            alpha=self.plot_alpha_fill,
            color=self.plot_color_band,
            label="mean ± std",
        )
        ax_std.plot(x, mean_y, color=self.plot_color_mean, linewidth=2, label="mean")
        ax_std.plot(
            x, mean_y + std_y, color=self.plot_color_band, linewidth=0.8, linestyle="--"
        )
        ax_std.plot(
            x, mean_y - std_y, color=self.plot_color_band, linewidth=0.8, linestyle="--"
        )
        ax_std.set_title(f"{agg_title} – {agg_series}  [mean ± std]")
        ax_std.set_xlabel("step")
        ax_std.set_ylabel("value")
        ax_std.legend(loc="best")
        ax_std.grid(True, alpha=0.3)
        fig_std.tight_layout()

        t_log.report_matplotlib_figure(
            title=f"{agg_title}/{agg_series}_mean-std",
            series="filled_plot",
            figure=fig_std,
            report_image=True,
        )
        plt.close(fig_std)

        # Plot B: mean / min / max
        fig_mm, ax_mm = plt.subplots(figsize=self.plot_fig_size)
        ax_mm.fill_between(
            x,
            min_y,
            max_y,
            alpha=self.plot_alpha_fill,
            color=self.plot_color_band,
            label="min – max range",
        )
        ax_mm.plot(x, mean_y, color=self.plot_color_mean, linewidth=2, label="mean")
        ax_mm.plot(
            x,
            max_y,
            color=self.plot_color_band,
            linewidth=0.8,
            linestyle="--",
            label="max",
        )
        ax_mm.plot(
            x,
            min_y,
            color=self.plot_color_band,
            linewidth=0.8,
            linestyle=":",
            label="min",
        )
        ax_mm.set_title(f"{agg_title} – {agg_series}  [mean / min / max]")
        ax_mm.set_xlabel("step")
        ax_mm.set_ylabel("value")
        ax_mm.legend(loc="best")
        ax_mm.grid(True, alpha=0.3)
        fig_mm.tight_layout()

        t_log.report_matplotlib_figure(
            title=f"{agg_title}/{agg_series}_mean-min-max",
            series="filled_plot",
            figure=fig_mm,
            report_image=True,
        )
        plt.close(fig_mm)


# ---------------------------------------------------------------------------
# RUNNING SCRIPT IN CLI
# ---------------------------------------------------------------------------


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
