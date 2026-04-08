import logging
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from clearml import Task, TaskTypes, Logger

from helpers.data_getter import DataGetter, SummaryType


class Summarizer:
    def __init__(
        self,
        tasks_data: DataGetter,
        summary_task_name: str = "SUMMARY",
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
            "matplotlib" – raster PNG stored in DEBUG SAMPLES tab
            "plotly"     – interactive vector figure stored in PLOTS tab
        plot_merged_metrics - Instead of calculating means, plot every task on the plot
        """

        self.log = logging.getLogger(__name__)
        self.plots_backend = plots_backend
        self.plot_merged_metrics = plot_merged_metrics
        self.merged_metrics = None

        self.tasks = tasks_data.tasks
        self.metric_paths = tasks_data.metrics_paths
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
        )

        self.log.info(
            f"Summary task '{self.project_name}/{self.summary_task_name}' (id={summary_task.task_id}) created."
        )

        if cleanup_previous_tags:
            self._cleanup_previous_tags()

        if add_tag_to_tasks:
            tags = [f"{tag_for_tasks}:{summary_task.task_id}"]
            for t_id in self.tasks.keys():
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
            summary_task.close()

        self.log.info(
            f"Finished summarization of {len(self.tasks)} tasks with {len(self.metric_paths)} metrics."
        )

    def _cleanup_previous_tags(self) -> None:
        self.log.info("Removing previous summary tags from task(s).")

        cleaned_tasks = 0
        removed_total = 0

        # TODO make it multithread!
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

    # TODO somehow make it palaller (multiprocessing?)
    def _summarize(self, summary_task: Task) -> None:
        something_failed = False

        metric_paths = self.metric_paths
        if self.plot_merged_metrics:
            self._merge_metrics_series()
            metric_paths = list(self.merged_metrics.keys())

        for cnt, path_str in enumerate(metric_paths):
            self.log.info(
                f"[METRIC {cnt + 1}/{len(metric_paths)}][START]:\t {path_str}"
            )

            if not self.plot_merged_metrics:
                series_y, axis_x = self._extract_series(path_str)

                if series_y is None:
                    self.log.warning(f"Skipping '{path_str}': extraction failed.")
                    self.log.debug(
                        f"The content of series_y (len: {len(series_y)}):\t {series_y}"
                    )
                    something_failed = True
                    continue

                data = self._calculate_series_summary(series_y)

            else:
                data, axis_x = self._extract_series_summary(path_str)

            summary_logger = summary_task.get_logger()
            self._report_scalars(summary_logger, axis_x, data, path_str)
            self._report_filled_plots(summary_logger, axis_x, data, path_str)

            self.log.info(f"[METRIC {cnt + 1}/{len(metric_paths)}][END]:\t {path_str}")

        if something_failed:
            self.log.warning(
                ">>>>>>>>>> SOMETHING WENT WRONG. CHECK THE LOGS! <<<<<<<<<<"
            )

    def _merge_metrics_series(self) -> dict[str, dict[str]]:
        self.merged_metrics = {}

        for path_str in self.metric_paths:
            path_parts = path_str.split("/")
            # top_key = everything except the last 2 parts (stat + "y")
            # e.g. "Loss/learning_rate/mean/y" → "Loss/learning_rate"
            top_key = "/".join(path_parts[:-2])

            if top_key not in self.merged_metrics:
                self.merged_metrics[top_key] = {}

            stat = path_parts[-2]
            self.merged_metrics[top_key][stat] = path_str

    def _extract_series(
        self,
        path_str: str,
    ) -> tuple[list[list[float]], list[float]]:
        path_parts = path_str.split("/")
        top_key = "/".join(path_parts[:-2])
        series_y: list[list[float]] = []
        axis_x: list[float] = None

        for t_data in self.tasks.values():
            metric = t_data[top_key]["series"]
            series_y.append(list(metric["y"]))

            if axis_x is None:
                axis_x = list(metric["x"])

        return series_y, axis_x

    def _calculate_series_summary(series: list[float]) -> dict[str, float]:
        summary = {}
        summary["mean"] = np.mean(series, axis=0)
        summary["std+"] = np.std(series, axis=0)
        summary["min"] = np.min(series, axis=0)
        summary["max"] = np.max(series, axis=0)
        return summary

    def _extract_series_summary(
        self,
        top_key: str,
    ) -> tuple[dict[str, dict[str, list[float]]], list[float]]:

        summary = {}
        axis_x: list[float] = None

        for stat, path_str in self.merged_metrics[top_key].items():
            path_parts = path_str.split("/")
            top_key = "/".join(path_parts[:-2])

            summary[stat] = {}
            for t_id, t_data in self.tasks.items():
                metric = t_data[top_key][stat]
                summary[stat][t_id] = list(metric["y"])

                if axis_x is None:
                    axis_x = list(range(0, len(metric["y"])))

        return summary, axis_x

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
        data: dict,
        path_str: str,
    ) -> None:
        """
        Upload summary to a ClearML task via its logger.
        """
        title, _ = self._path_to_title_series(path_str)

        mean_y = data["mean"]
        std_y = data["std+"]
        min_y = data["min"]
        max_y = data["max"]

        for step in range(len(axis_x)):
            for t_id, mean in mean_y.items():
                x = axis_x[step]

                if SummaryType.MEAN in self.summary_types:
                    # t_log.report_scalar(f"{title}_mean", "mean", mean_y[step], x)
                    t_log.report_scalar(f"{title}_mean", t_id, mean[step], x)

                if SummaryType.MEAN_MINMAX in self.summary_types:
                    t_log.report_scalar(
                        f"{title}_mean-min-max", "mean", mean_y[step], x
                    )
                    t_log.report_scalar(f"{title}_mean-min-max", "min", min_y[step], x)
                    t_log.report_scalar(f"{title}_mean-min-max", "max", max_y[step], x)

                if SummaryType.MEAN_STD in self.summary_types:
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
        if SummaryType.MEAN_STD in self.summary_types:
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
        if SummaryType.MEAN_MINMAX in self.summary_types:
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

        # Plot C: mean
        if SummaryType.MEAN in self.summary_types:
            fig_mm = go.Figure()

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
                title=f"{agg_title} – {agg_series}  [mean]",
                xaxis_title="step",
                yaxis_title="value",
            )

            t_log.report_plotly(
                title=f"{agg_title}_mean",
                series=f"{agg_title}_mean",
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
        if SummaryType.MEAN_STD in self.summary_types:
            fig_std, ax_std = plt.subplots(figsize=self.plot_fig_size)
            ax_std.fill_between(
                x,
                mean_y - std_y,
                mean_y + std_y,
                alpha=self.plot_alpha_fill,
                color=self.plot_color_band,
                label="mean ± std",
            )
            ax_std.plot(
                x, mean_y, color=self.plot_color_mean, linewidth=2, label="mean"
            )
            ax_std.plot(
                x,
                mean_y + std_y,
                color=self.plot_color_band,
                linewidth=0.8,
                linestyle="--",
            )
            ax_std.plot(
                x,
                mean_y - std_y,
                color=self.plot_color_band,
                linewidth=0.8,
                linestyle="--",
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
        if SummaryType.MEAN_MINMAX in self.summary_types:
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

        # Plot C: mean
        if SummaryType.MEAN in self.summary_types:
            fig_mm, ax_mm = plt.subplots(figsize=self.plot_fig_size)
            ax_mm.plot(x, mean_y, color=self.plot_color_mean, linewidth=2, label="mean")
            ax_mm.set_title(f"{agg_title} – {agg_series}  [mean]")
            ax_mm.set_xlabel("step")
            ax_mm.set_ylabel("value")
            ax_mm.legend(loc="best")
            ax_mm.grid(True, alpha=0.3)
            fig_mm.tight_layout()

            t_log.report_matplotlib_figure(
                title=f"{agg_title}/{agg_series}_mean",
                series="filled_plot",
                figure=fig_mm,
                report_image=True,
            )
            plt.close(fig_mm)
