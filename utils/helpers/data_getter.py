import copy
import logging
from enum import StrEnum
from typing import Iterable, Optional

from clearml import Task
from clearml.backend_api.services import projects as projects_service
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm


class SummaryType(StrEnum):
    MEAN = "mean"
    MEAN_MINMAX = "mean-min-max"
    MEAN_STD = "mean-std"


class DataGetter:
    def __init__(
        self,
        project_name: str,
        tags_select: Iterable[str],
        recursive_projects: bool = False,
        metrics_paths: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
        merge_summaries_metrics: bool = False,
    ):

        self.log = logging.getLogger(__name__)
        self.project_name = project_name
        self.max_samples = max_samples
        self.recursive_projects = recursive_projects

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
            # TODO implement merge of the summaries for selected metrics
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
        remove_colon_scalars: bool = True,
        same_n_iterations: bool = True,
    ) -> dict[str, dict]:
        # Returns: dict[task_id, task_data_dict]

        tasks = {}
        tasks_raw = self._get_cleraml_tasks()
        if tasks_raw is None:
            return tasks

        self.log.info("Validating tasks metrics.")
        self.min_n_iterations = tasks_raw[0].get_last_iteration()

        with logging_redirect_tqdm():
            for t in tqdm(tasks_raw):
                try:
                    data = self._task_data_getter(t)
                    if remove_colon_scalars:
                        data = self._remove_colcon_scalars(data)

                    if same_n_iterations:
                        self._compare_n_iterations(t.get_last_iteration())

                    tasks[t.task_id] = data
                except ValueError as e:
                    self.log.warning(f"Skipping task id {t.task_id} due to: {str(e)}")
        return tasks

    def _get_cleraml_tasks(self) -> list[Task]:
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
        return None

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
        self.log.info("Auto-detecting shared metric path(s).")
        metric_paths = self._detect_common_metric_paths()
        if not metric_paths:
            self.log.error(
                "No scalar metrics are shared by all selected tasks. Returning empty list."
            )
            return []
        if self.merge_summaries_metrics:
            self.log.info(
                f"Found {len(metric_paths)} metric path(s). Merging already processed metrics via title suffixes."
            )
            metric_paths = self._merge_metrics_series(metric_paths)

        projects_str = "\n\t".join(metric_paths)
        self.log.info(
            f"Auto-detected {len(metric_paths)} shared metric path(s)\n\t{projects_str}"
        )

        return metric_paths

    def _detect_common_metric_paths(
        self,
        skip_time_series: bool = True,
    ) -> list[str]:
        # We need a shallow copy to modify the original dictionary
        tasks = copy.copy(self.tasks)
        path_sets: list[set[str]] = [None] * len(tasks)

        with logging_redirect_tqdm():
            for cnt, (t_id, t_data) in tqdm(enumerate(tasks.items())):
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

                path_sets[cnt] = paths

        if not path_sets:
            self.log.error("The given list of tasks do not have any metrics.")
            return []

        common_paths = path_sets[0].intersection(*path_sets[1:])
        return sorted(common_paths)

    def _merge_metrics_series(self, metric_paths: list[str]) -> list[str]:

        results: set[str] = set()

        with logging_redirect_tqdm():
            for t_id in tqdm(list(self.tasks.keys())):
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
                        y: self.tasks[t_id][m_type_title][series][y]
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
