import colorsys
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from clearml import Task, TaskTypes, Logger
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from helpers.data_getter import DataGetter, SummaryType


@dataclass(slots=True)
class StatisticsData:
    series_name: str
    data_num: int
    mean: list[float]
    std: list[float]
    minimum: list[float]
    maximum: list[float]


@dataclass(slots=True)
class StatisticsSeries:
    name: str
    series: list[StatisticsData]


FIELD_MAP = {"mean": "mean", "std": "std", "min": "minimum", "max": "maximum"}
DEFAULT_SUMMARY_TASK_NAME = "SUMMARY"


class Summarizer:
    def __init__(
        self,
        tasks_data: DataGetter,
        summary_task_name: str = DEFAULT_SUMMARY_TASK_NAME,
        summary_task_tags: list[str] = ["summary"],
        summary_types: list[SummaryType] = [
            SummaryType.MEAN_STD,
            SummaryType.MEAN_MINMAX,
        ],
        plots_backend: Literal["matplotlib", "plotly", "None"] = "plotly",
        plot_merged_metrics: bool = False,
    ):
        """
        plots_backend:
            "matplotlib" - raster PNG stored in DEBUG SAMPLES tab
            "plotly"     - interactive vector figure stored in PLOTS tab
        plot_merged_metrics - Instead of calculating means, plot every task on the plot
        """

        self.log = logging.getLogger(__name__)
        self.plots_backend = plots_backend
        self.enable_summary_processing = plot_merged_metrics

        self.tasks = tasks_data.tasks
        self.metric_paths = tasks_data.metrics_paths
        self.x_axis = self._get_x_axis()
        self.metrics_stats = self._process_metrics_statistics()

        self.project_name = tasks_data.project_name
        self.i_tags_filter = tasks_data.tags_filter
        self.i_max_samples = tasks_data.max_samples

        self.summary_task_name = summary_task_name
        self.summary_task_tags = summary_task_tags
        self.summary_task_type = TaskTypes.application
        self.summary_types = set(summary_types)

        self.plot_alpha_fill = 0.25
        self.plot_color_mean = "#2271B5"
        self.plot_color_band = "#6BAED6"
        self.plot_fig_size = (9, 4)

        self.log.info(f"Using `{self.plots_backend}` for plots backend.")

    def _get_x_axis(self) -> Optional[list[float]]:
        self.log.info("Obtaining X axis")

        x_axis = next(
            (
                series["x"]
                for t_data in self.tasks.values()
                for metrics in t_data.values()
                for series in metrics.values()
                if "x" in series
            ),
            None,
        )

        if x_axis is not None:
            self.log.info(f"Got the X axis definition with {len(x_axis)} steps.")
            return x_axis

        self.log.warning(
            "Failed to get the X axis, the length of series will be used instead."
        )
        y_axis = next(
            (
                series["y"]
                for t_data in self.tasks.values()
                for metrics in t_data.values()
                for series in metrics.values()
                if "y" in series
            ),
            None,
        )
        return [x for x in range(len(y_axis))]

    def _process_metrics_statistics(self) -> dict[str, StatisticsSeries]:
        if self.enable_summary_processing:
            self.log.info(
                f"Extracting statistical summary data from {len(self.tasks)} tasks."
            )
            return self._process_merged_metrics()

        self.log.info(
            f"Proceeding to stasticial summarization of {len(self.tasks)} tasks for {len(self.metric_paths)} metrics."
        )
        return self._process_summarized_metrics()

    def _process_merged_metrics(self) -> dict[str, StatisticsSeries]:
        """Read pre-processed per-task data directly — no statistics calculation."""
        metrics_stats: dict[str, StatisticsSeries] = {}

        # Group stats by top-level metric key
        metrics: dict[str, set[str]] = {}
        for path_str in self.metric_paths:
            parts = path_str.split("/")
            top_key = "/".join(parts[:-2])
            stat = parts[-2]
            metrics.setdefault(top_key, set()).add(stat)

        self.log.info(f"Merged metrics paths into {len(metrics)} top-level metrics.")
        self.log.info(f"Extracting summary data from {len(self.tasks)} tasks.")

        with logging_redirect_tqdm():
            for cnt, (top_key, stats) in tqdm(
                enumerate(metrics.items()),
                total=len(metrics),
                desc="Metric",
                leave=False,
                position=0,
            ):
                self.log.info(f"[METRIC {cnt + 1}/{len(metrics)}]:\t {top_key}")
                n_tasks = len(self.tasks)
                metrics_stats[top_key] = StatisticsSeries(
                    name=top_key, series=[None] * n_tasks
                )

                unique_series_names = set()
                for idx, (t_id, t_data) in enumerate(self.tasks.items()):
                    m_stats = {
                        FIELD_MAP[stat]: t_data[top_key][stat]["y"] for stat in stats
                    }

                    task = Task.get_task(task_id=t_id)
                    series_name: str = task.name

                    if series_name == DEFAULT_SUMMARY_TASK_NAME:
                        series_name = Path(task.get_project_name()).name
                    elif series_name.startswith(f"{DEFAULT_SUMMARY_TASK_NAME}_"):
                        series_name = series_name.removeprefix(
                            f"{DEFAULT_SUMMARY_TASK_NAME}_"
                        )

                    if series_name in unique_series_names:
                        series_name = f"{unique_series_names}_{t_id[:6]}"

                    unique_series_names.add(series_name)
                    metrics_stats[top_key].series[idx] = StatisticsData(
                        series_name=series_name,
                        data_num=None,  # TODO: somehow restore data_num info from previous summaries (from the series name)
                        **m_stats,
                    )

            self.log.info("Finished statistical summarization.")

            return metrics_stats

    def _process_summarized_metrics(self) -> dict[str, StatisticsSeries]:
        """Aggregate raw series from all tasks into mean/std/min/max."""
        metrics_stats: dict[str, StatisticsSeries] = {}

        with logging_redirect_tqdm():
            for cnt, path_str in tqdm(
                enumerate(self.metric_paths), desc="Metric", leave=False, position=0
            ):
                parts = path_str.split("/")
                top_key = "/".join(parts[:-2])
                stat = parts[-2]
                variable = parts[-1]

                self.log.info(
                    f"[METRIC {cnt + 1}/{len(self.metric_paths)}]:\t {path_str}"
                )

                metrics_stats.setdefault(
                    top_key, StatisticsSeries(name=top_key, series=[None])
                )

                data_series = [
                    list(t_data[top_key][stat][variable])
                    for t_data in self.tasks.values()
                ]

                if not data_series:
                    self.log.warning(f"Skipping '{path_str}': extraction failed.")
                    continue

                metrics_stats[top_key].series[0] = StatisticsData(
                    series_name=f"tasks_n{len(self.tasks)}",
                    data_num=len(self.tasks),
                    mean=np.mean(data_series, axis=0),
                    std=np.std(data_series, axis=0),
                    minimum=np.min(data_series, axis=0),
                    maximum=np.max(data_series, axis=0),
                )
            self.log.info("Finished extraction of statistical summaries.")

            return metrics_stats

    def summarize(
        self,
        add_tag_to_tasks: bool = True,
        tag_for_tasks: str = "summary",
        cleanup_previous_tags: bool = True,
    ) -> None:
        summary_task = Task.init(
            project_name=self.project_name,
            task_name=self.summary_task_name,
            task_type=self.summary_task_type,
            tags=self.summary_task_tags,
            reuse_last_task_id=False,
            auto_resource_monitoring=False,
            auto_connect_streams=False,
        )

        self.log.info(
            f"Summary task '{self.project_name}/{self.summary_task_name}' (id={summary_task.task_id}) created."
        )

        if cleanup_previous_tags:
            self._cleanup_previous_tags()

        if add_tag_to_tasks:
            tags = [f"{tag_for_tasks}:{summary_task.task_id}"]
            self.log.info("Adding tag(s) to the source tasks.")
            with logging_redirect_tqdm():
                for t_id in tqdm(
                    self.tasks.keys(), desc="Task", leave=False, position=0
                ):
                    t = Task.get_task(task_id=t_id)
                    t.add_tags(tags)
            self.log.info(f"Added tag(s) {tags} to {len(self.tasks)} source tasks.")

        # Persist configuration for reproducibility
        summary_task.set_parameter("summarize/tags_filter", str(self.i_tags_filter))
        summary_task.set_parameter("summarize/metric_paths", str(self.metric_paths))
        summary_task.set_parameter("summarize/max_samples", self.i_max_samples)
        summary_task.set_parameter("summarize/n_source_tasks", len(self.tasks))

        try:
            self._summarize(summary_task)
        except Exception as e:
            self.log.exception(f"Caught an exception: {e}")
        finally:
            self.log.info("Closing the summary task.")
            summary_task.close()

        self.log.info(
            f"Finished summarization of {len(self.tasks)} tasks with {len(self.metric_paths)} metrics."
        )

    def _cleanup_previous_tags(self) -> None:
        self.log.info("Removing previous summary tags from task(s).")

        cleaned_tasks = 0
        removed_total = 0

        # TODO make it multithread!
        with logging_redirect_tqdm():
            for t_id in tqdm(self.tasks.keys(), desc="Task", leave=False, position=0):
                t = Task.get_task(task_id=t_id)

                current_tags = list(t.get_tags() or [])
                filtered_tags = [
                    tag for tag in current_tags if not tag.startswith("summary:")
                ]

                if filtered_tags != current_tags:
                    removed = [
                        tag for tag in current_tags if tag.startswith("summary:")
                    ]
                    t.set_tags(filtered_tags)
                    cleaned_tasks += 1
                    removed_total += len(removed)
                    self.log.debug(
                        f"Removed summary tags from task id {t_id}: {removed}"
                    )

        self.log.info(
            f"Removed {removed_total} summary tag(s) from {cleaned_tasks} task(s)."
        )

    # TODO somehow make it palaller (multiprocessing?)
    def _summarize(self, summary_task: Task) -> None:

        with logging_redirect_tqdm():
            for cnt, metric in tqdm(
                enumerate(self.metrics_stats.values()),
                total=len(self.metrics_stats),
                desc="Metric",
                leave=False,
                position=0,
            ):
                self.log.info(
                    f"[METRIC {cnt + 1}/{len(self.metrics_stats)}]:\t {metric.name}"
                )

                summary_logger = summary_task.get_logger()
                self._report_scalars(summary_logger, metric)
                self._report_filled_plots(summary_logger, metric)

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
        metric: StatisticsSeries,
    ) -> None:
        """
        Upload summary to a ClearML task via its logger.
        """
        self.log.info("Reporting scalars to ClearML server.")
        title = metric.name

        for step in tqdm(
            range(len(self.x_axis)), desc="Step..", leave=False, position=1
        ):
            x = self.x_axis[step]
            for series in metric.series:
                series_name = series.series_name
                mean = series.mean
                std = series.std
                minimum = series.minimum
                maximum = series.maximum

                mean_name = "mean" if len(metric.series) == 1 else series_name
                prefix = "" if len(metric.series) == 1 else f"{series_name}_"

                if SummaryType.MEAN in self.summary_types:
                    t_log.report_scalar(f"{title}_mean", mean_name, mean[step], x)

                if SummaryType.MEAN_MINMAX in self.summary_types:
                    t_log.report_scalar(
                        f"{title}_mean-min-max", f"{prefix}mean", mean[step], x
                    )
                    t_log.report_scalar(
                        f"{title}_mean-min-max",
                        f"{prefix}min",
                        minimum[step],
                        x,
                    )
                    t_log.report_scalar(
                        f"{title}_mean-min-max",
                        f"{prefix}max",
                        maximum[step],
                        x,
                    )

                if SummaryType.MEAN_STD in self.summary_types:
                    t_log.report_scalar(
                        f"{title}_mean-std", f"{prefix}mean", mean[step], x
                    )
                    t_log.report_scalar(
                        f"{title}_mean-std",
                        f"{prefix}std+",
                        mean[step] + std[step],
                        x,
                    )
                    t_log.report_scalar(
                        f"{title}_mean-std",
                        f"{prefix}std-",
                        mean[step] - std[step],
                        x,
                    )

    def _report_filled_plots(
        self,
        t_log: Logger,
        metric: StatisticsSeries,
    ) -> None:
        """
        Create filled confidence-band figures and upload them to a ClearML task.
        """
        self.log.info("Creating and uploading plots to ClearML server.")

        match self.plots_backend:
            case "plotly":
                self._report_filled_plots_plotly(t_log=t_log, metric=metric)
            case "matplotlib":
                self._report_filled_plots_matplotlib(t_log=t_log, metric=metric)
            case "None":
                pass
            case _:
                self.log.warning(
                    f"Unrecognized plots backend `{self.plots_backend}` "
                    "(Available: `plotly`, `matplotlib`). Skipping plots."
                )

    def _report_filled_plots_plotly(
        self,
        t_log: Logger,
        metric: StatisticsSeries,
    ) -> None:

        series_colors = self._generate_series_colors(
            n=len(metric.series),
            single_line_color=self.plot_color_mean,
            single_band_color=self.plot_color_band,
            band_alpha=self.plot_alpha_fill,
        )

        x = np.asarray(self.x_axis)
        metric_name = metric.name

        # One figure per summary type, all series added before reporting
        fig_mean = go.Figure() if SummaryType.MEAN in self.summary_types else None
        fig_mm = go.Figure() if SummaryType.MEAN_MINMAX in self.summary_types else None
        fig_std = go.Figure() if SummaryType.MEAN_STD in self.summary_types else None

        for series, (line_color, fill_color) in tqdm(
            zip(metric.series, series_colors), desc="Series", leave=False, position=1
        ):
            series_name = series.series_name
            mean = np.asarray(series.mean)
            std = np.asarray(series.std)
            minimum = np.asarray(series.minimum)
            maximum = np.asarray(series.maximum)

            if fig_mean is not None:
                fig_mean.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        line=dict(color=line_color, width=2),
                        hovertemplate="mean: %{y:.4f}<extra></extra>",
                        name=series_name,
                    )
                )

            if fig_mm is not None:
                fig_mm.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([maximum, minimum[::-1]]),
                        fill="toself",
                        fillcolor=fill_color,
                        line=dict(color=line_color),
                        hoverinfo="skip",
                        name=f"{series_name}-range",
                    )
                )
                fig_mm.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        line=dict(color=line_color, width=2),
                        hovertemplate="mean: %{y:.4f}<extra></extra>",
                        name=series_name,
                    )
                )

            if fig_std is not None:
                fig_std.add_trace(
                    go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([mean + std, (mean - std)[::-1]]),
                        fill="toself",
                        fillcolor=fill_color,
                        line=dict(color=line_color),
                        hoverinfo="skip",
                        name=f"{series_name}-std",
                    )
                )
                fig_std.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean,
                        line=dict(color=line_color, width=2),
                        hovertemplate="mean: %{y:.4f}<extra></extra>",
                        name=series_name,
                    )
                )

        # Report once per summary type, after all series are added
        if fig_mean is not None:
            fig_mean.update_layout(
                title=f"{metric_name} [mean]",
                xaxis_title="step",
                yaxis_title="value",
            )
            t_log.report_plotly(
                title=f"{metric_name}_mean",
                # series=f"{metric_name}_mean",
                series="mean",
                figure=fig_mean,
            )

        if fig_mm is not None:
            fig_mm.update_layout(
                title=f"{metric_name} [mean / min / max]",
                xaxis_title="step",
                yaxis_title="value",
            )
            t_log.report_plotly(
                title=f"{metric_name}_mean-min-max",
                # series=f"{metric_name}_mean-min-max",
                series="mean-min-max",
                figure=fig_mm,
            )

        if fig_std is not None:
            fig_std.update_layout(
                title=f"{metric_name} [mean ± std]",
                xaxis_title="step",
                yaxis_title="value",
            )
            t_log.report_plotly(
                title=f"{metric_name}_mean-std",
                # series=f"{metric_name}_mean-std",
                series="mean-std",
                figure=fig_std,
            )

    def _report_filled_plots_matplotlib(
        self,
        t_log: Logger,
        metric: StatisticsSeries,
    ) -> None:
        x = np.asarray(self.x_axis)
        metric_name = metric.name

        # Create one figure per summary type upfront
        fig_mean, ax_mean = (
            plt.subplots(figsize=self.plot_fig_size)
            if SummaryType.MEAN in self.summary_types
            else (None, None)
        )
        fig_mm, ax_mm = (
            plt.subplots(figsize=self.plot_fig_size)
            if SummaryType.MEAN_MINMAX in self.summary_types
            else (None, None)
        )
        fig_std, ax_std = (
            plt.subplots(figsize=self.plot_fig_size)
            if SummaryType.MEAN_STD in self.summary_types
            else (None, None)
        )

        for series in tqdm(metric.series, desc="Series", leave=False, position=1):
            series_name = series.series_name
            mean = np.asarray(series.mean)
            std = np.asarray(series.std)
            minimum = np.asarray(series.minimum)
            maximum = np.asarray(series.maximum)

            if ax_mean is not None:
                ax_mean.plot(x, mean, linewidth=2, label=series_name)

            if ax_mm is not None:
                ax_mm.fill_between(
                    x,
                    minimum,
                    maximum,
                    alpha=self.plot_alpha_fill,
                    label=f"{series_name}-range",
                )
                ax_mm.plot(x, mean, linewidth=2, label=series_name)
                ax_mm.plot(x, maximum, linewidth=0.8, linestyle="--")
                ax_mm.plot(x, minimum, linewidth=0.8, linestyle=":")

            if ax_std is not None:
                ax_std.fill_between(
                    x,
                    mean - std,
                    mean + std,
                    alpha=self.plot_alpha_fill,
                    label=f"{series_name}-std",
                )
                ax_std.plot(x, mean, linewidth=2, label=series_name)
                ax_std.plot(x, mean + std, linewidth=0.8, linestyle="--")
                ax_std.plot(x, mean - std, linewidth=0.8, linestyle="--")

        # Finalise and report once per summary type
        if fig_mean is not None:
            ax_mean.set_title(f"{metric_name} [mean]")
            ax_mean.set_xlabel("step")
            ax_mean.set_ylabel("value")
            ax_mean.legend(loc="best")
            ax_mean.grid(True, alpha=0.3)
            fig_mean.tight_layout()
            t_log.report_matplotlib_figure(
                title=f"{metric_name}_mean",
                series="filled_plot",
                figure=fig_mean,
                report_image=True,
            )
            plt.close(fig_mean)

        if fig_mm is not None:
            ax_mm.set_title(f"{metric_name} [mean / min / max]")
            ax_mm.set_xlabel("step")
            ax_mm.set_ylabel("value")
            ax_mm.legend(loc="best")
            ax_mm.grid(True, alpha=0.3)
            fig_mm.tight_layout()
            t_log.report_matplotlib_figure(
                title=f"{metric_name}_mean-min-max",
                series="filled_plot",
                figure=fig_mm,
                report_image=True,
            )
            plt.close(fig_mm)

        if fig_std is not None:
            ax_std.set_title(f"{metric_name} [mean ± std]")
            ax_std.set_xlabel("step")
            ax_std.set_ylabel("value")
            ax_std.legend(loc="best")
            ax_std.grid(True, alpha=0.3)
            fig_std.tight_layout()
            t_log.report_matplotlib_figure(
                title=f"{metric_name}_mean-std",
                series="filled_plot",
                figure=fig_std,
                report_image=True,
            )
            plt.close(fig_std)

    @staticmethod
    def _generate_series_colors(
        n: int,
        single_line_color: str,
        single_band_color: str,
        band_alpha: float,
    ) -> list[tuple[str, str]]:
        """
        Returns a list of (line_rgba, fill_rgba) pairs — one per series.
        For n==1 the class defaults are used as-is.
        """
        if n == 1:
            hex_color = single_line_color.lstrip("#")
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
            line = f"rgb({r},{g},{b})"
            fill = f"rgba({r},{g},{b},{band_alpha})"
            return [(line, fill)]

        colors = []
        for i in range(n):
            hue = i / n  # evenly spaced hues around the wheel
            r_f, g_f, b_f = colorsys.hsv_to_rgb(hue, 0.75, 0.80)  # vivid line color
            r, g, b = int(r_f * 255), int(g_f * 255), int(b_f * 255)

            # Lighter fill: lower saturation + higher value
            r_l, g_l, b_l = colorsys.hsv_to_rgb(hue, 0.35, 0.95)
            rl, gl, bl = int(r_l * 255), int(g_l * 255), int(b_l * 255)

            line = f"rgb({r},{g},{b})"
            fill = f"rgba({rl},{gl},{bl},{band_alpha})"
            colors.append((line, fill))

        return colors
