import functools
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Optional

from clearml import Task
from clearml.backend_api.services import projects as projects_service
from joblib import Parallel, delayed


class NoTasksError(Exception):
    pass


class NoMetricsError(Exception):
    pass


class SummaryType(StrEnum):
    MEAN = "mean"
    MEAN_MINMAX = "mean-min-max"
    MEAN_STD = "mean-std"


@dataclass(slots=True)
class TaskLoadResult:
    task_id: str
    last_iteration: int | None
    data: any | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


N_JOBS = -1  # use all available cores/threads


class DataGetter:
    def __init__(
        self,
        project_name: str,
        tags_select: Iterable[str],
        recursive_projects: bool = False,
        metrics_paths: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
        merge_summaries_metrics: bool = False,
        n_jobs: int = N_JOBS,
    ):

        self.log = logging.getLogger(__name__)
        self.project_name = project_name
        self.max_samples = max_samples
        self.recursive_projects = recursive_projects
        self.n_jobs = n_jobs

        self.merge_summaries_metrics = merge_summaries_metrics
        self.MERGE_SERIES_MAPPING = {
            "std+": "std",
            "std-": None,
        }
        if self.merge_summaries_metrics:
            self.log.info("Merging summaries: ENABLED")
            self.log.debug(
                f"Mapping for merging series (`None` drops data): {self.MERGE_SERIES_MAPPING}"
            )

        self.CLEARML_MAX_GET_TASKS = 500
        self.CLEARML_TASKS_FILTER = {
            "status": ["completed"],
            "type": ["training", "testing", "inference", "application"],
            "system_tags": ["-archived"],
        }
        self.log.debug(f"Task filter options: {self.CLEARML_TASKS_FILTER}")

        self.tags_raw = tags_select
        self.tags_filter = self._prepare_tags_filter(tags_select)
        self.tasks: dict[str, dict] = self._create_tasks_data_dict()
        if metrics_paths is None:
            metrics_paths = self._get_common_metrics()
        else:
            self._filter_tasks_and_select_metrics(metrics_paths)
        self.metrics_paths = metrics_paths

        self.assert_data_existence()
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
        remove_colon_scalars: bool = True,
        same_n_iterations: bool = True,
    ) -> dict[str, dict]:
        tasks = {}
        tasks_raw = self._get_clearml_tasks()
        if not tasks_raw:
            return tasks

        self.log.info("Validating tasks metrics.")

        with Parallel(n_jobs=self.n_jobs, backend="threading") as parallel:
            results: list[TaskLoadResult] = parallel(
                delayed(self._load_one_task)(t, remove_colon_scalars) for t in tasks_raw
            )

        valid_iterations = []
        for result in results:
            if not result.ok:
                self.log.warning(
                    f"Skipping task id {result.task_id} due to: {result.error}"
                )
                continue
            tasks[result.task_id] = result.data
            if same_n_iterations and result.last_iteration is not None:
                valid_iterations.append(result.last_iteration)

        if same_n_iterations and valid_iterations:
            self.min_n_iterations = min(valid_iterations)

            tasks = {
                result.task_id: result.data
                for result in results
                if result.ok and result.last_iteration >= self.min_n_iterations
            }

        return tasks

    def _load_one_task(self, t: Task, remove_colon_scalars: bool) -> TaskLoadResult:
        try:
            data = self._task_data_getter(t)
            if remove_colon_scalars:
                data = self._remove_colon_prefixed_scalars(data)
            return TaskLoadResult(
                task_id=t.task_id,
                last_iteration=t.get_last_iteration(),
                data=data,
            )
        except ValueError as e:
            return TaskLoadResult(
                task_id=t.task_id, last_iteration=None, data=None, error=str(e)
            )

    def _get_clearml_tasks(self) -> list[Task]:
        self.log.info(
            f"Querying tasks with tags {self.tags_raw} (AND) in project '{self.project_name}' (Recursive: {self.recursive_projects})."
        )

        project_names = [self.project_name]
        if self.recursive_projects:
            project_names = self._get_recursive_project_names(self.project_name)
            projects_str = "\n\t".join(project_names)
            self.log.info(
                f"Found {len(project_names)} recursive project(s) to scan:\n\t{projects_str}"
            )

        tasks = Task.get_tasks(
            project_name=project_names,
            tags=self.tags_filter,
            task_filter=self.CLEARML_TASKS_FILTER,
        )

        if len(tasks) >= self.CLEARML_MAX_GET_TASKS:
            self.log.warning(
                f"Found maximum number of tasks ({self.CLEARML_MAX_GET_TASKS})!"
            )
        if tasks:
            self.log.info(f"Found {len(tasks)} task(s).")
            return tasks

        self.log.warning(
            f"No tasks found for tags {self.tags_filter} (AND) - nothing to summarise."
        )
        return []

    def _get_recursive_project_names(self, root_project: str) -> list[str]:
        """Return the root project plus all sub-projects under it."""

        session = Task._get_default_session()
        res = session.send(
            projects_service.GetAllRequest(
                # ClearML supports regex on project name
                name=f"^{root_project}(/.*)?$",
                only_fields=["name"],
                search_hidden=False,
            )
        )

        if not res or not res.response or not res.response.projects:
            self.log.warning(
                f"No projects found matching '{root_project}'. Using it as-is."
            )
            return [root_project]

        names = [p.name for p in res.response.projects]
        self.log.debug(f"Resolved project names: {names}")
        return names

    def _task_data_getter(self, t: Task) -> dict:
        if self.max_samples is None:
            return t.get_all_reported_scalars()
        return t.get_reported_scalars(max_samples=self.max_samples)

    def _remove_colon_prefixed_scalars(self, data: dict) -> dict:
        for k in list(data.keys()):
            if k.startswith(":"):
                data.pop(k)
        return data

    def _get_common_metrics(self) -> list[str]:
        self.log.info("Auto-detecting shared metric path(s).")
        metric_paths = self._filter_tasks_and_get_common_metric_paths()
        if not metric_paths:
            self.log.error(
                "No scalar metrics are shared by all selected tasks. Returning empty list."
            )
            return []
        if self.merge_summaries_metrics:
            self.log.info(
                f"Found {len(metric_paths)} metric path(s). Merging already processed metrics via title suffixes."
            )
            metric_paths = self._filter_tasks_and_get_merged_metrics_series(
                metric_paths
            )

        projects_str = "\n\t".join(metric_paths)
        self.log.info(
            f"Auto-detected {len(metric_paths)} shared metric path(s):\n\t{projects_str}"
        )

        return metric_paths

    # TODO(issue#91) Refactor function creep
    def _filter_tasks_and_get_common_metric_paths(
        self, skip_time_series: bool = True
    ) -> list[str]:
        path_sets = []

        to_remove = set()
        for t_id, t_data in self.tasks.items():
            paths = self._extract_task_paths(t_id, t_data, skip_time_series)
            if paths is None:
                self.log.warning(
                    f"Task {t_id} reported no scalar metrics. Omitting it from analysis."
                )
                to_remove.add(t_id)
                continue
            path_sets.append(paths)

        for t_id in to_remove:
            self.tasks.pop(t_id)

        if not path_sets:
            self.log.error("The given list of tasks do not have any metrics.")
            return []

        return sorted(path_sets[0].intersection(*path_sets[1:]))

    def _extract_task_paths(
        self, t_id: str, t_data: dict, skip_time_series: bool = True
    ) -> Optional[set[str]]:
        if not t_data:
            return None

        return {
            f"{title}/{series}/y"
            for title, series_dict in t_data.items()
            if not (skip_time_series and "time" in title.lower())
            for series in series_dict
        }

    # TODO(issue#91) Refactor function creep
    def _filter_tasks_and_get_merged_metrics_series(
        self, metric_paths: list[str]
    ) -> list[str]:

        results: set[str] = set()

        for t_id in list(self.tasks.keys()):
            titles_to_remove: set[str] = set()

            for metric in metric_paths:
                m_type, m_title, series, y = metric.split("/")
                new_series = self._map_series_names(series)
                if new_series is None:
                    continue

                m_type_title = f"{m_type}/{m_title}"
                m_title_merged = self._remove_strenum_suffix(m_title, SummaryType)
                m_type_title_merged = f"{m_type}/{m_title_merged}"

                if m_type_title_merged not in self.tasks[t_id]:
                    self.tasks[t_id][m_type_title_merged] = {}

                self.tasks[t_id][m_type_title_merged][new_series] = {
                    y: self.tasks[t_id][m_type_title][series][y],
                }

                results.add(f"{m_type_title_merged}/{new_series}/{y}")
                titles_to_remove.add(m_type_title)

            for m_type_title in titles_to_remove:
                self.tasks[t_id].pop(m_type_title)

        return sorted(results)

    def _map_series_names(self, series_name: str) -> Optional[str]:
        # Hardcoded mapping for storing new data. Returns None if there is need to drop the data.
        if not self.merge_summaries_metrics:
            return series_name
        return self.MERGE_SERIES_MAPPING.get(series_name, series_name)

    @staticmethod
    def _remove_strenum_suffix(s: str, enum: StrEnum) -> str:
        for item in enum:
            suffix = f"_{item.value}"
            if s.endswith(suffix):
                return s.removesuffix(suffix)
        return s

    # TODO(issue#91) Refactor function creep
    def _filter_tasks_and_select_metrics(self, metric_paths: list[str]):
        for path_str in metric_paths:
            path_parts = self._normalize_metric_path(path_str)

            to_remove = set()
            for t_id, t_data in self.tasks.items():
                if not t_data:
                    self.log.warning(
                        f"Task {t_id} reported no scalar metrics. Omitting it from analysis."
                    )
                    to_remove.add(t_id)
                    continue

                try:
                    functools.reduce(lambda d, key: d[key], path_parts, t_data)
                except KeyError:
                    self.log.warning(
                        f"Metric path '{path_str}' not found in the task id {t_id}. This task will be omitted.",
                    )
                    to_remove.add(t_id)

            for t_id in to_remove:
                self.tasks.pop(t_id)

    def _normalize_metric_path(self, path_str: str) -> list[str]:
        parts = path_str.split("/")
        return ["/".join(parts[:-2]), *parts[-2:]]

    def assert_data_existence(self) -> None:
        if not self.tasks:
            self.log.error("Failed to found valid tasks.")
            raise NoTasksError()

        if not self.metrics_paths:
            self.log.error("Failed to select any valid metrics.")
            raise NoMetricsError()
