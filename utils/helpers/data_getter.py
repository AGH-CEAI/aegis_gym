import copy
import logging
from typing import Iterable, Optional

from clearml import Task
from clearml.backend_api.services import projects as projects_service


class DataGetter:
    def __init__(
        self,
        project_name: str,
        tags_select: Iterable[str],
        recursive_projects: bool = False,
        metrics_paths: Optional[list[str]] = None,
        max_samples: Optional[int] = None,
    ):

        self.log = logging.getLogger(__name__)
        self.project_name = project_name
        self.max_samples = max_samples
        self.recursive_projects = recursive_projects

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

        self.min_n_iterations = tasks_raw[0].get_last_iteration()

        for t in tasks_raw:
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
        self.log.info("Auto-detecteding shared metric path(s).")
        metric_paths = self._detect_common_metric_paths()
        if not metric_paths:
            self.log.error(
                "No scalar metrics are shared by all selected tasks. Returning empty list."
            )
            return []

        projects_str = "\n\t".join(metric_paths)
        self.log.info(
            f"Auto-detected {len(metric_paths)} shared metric path(s)\n{projects_str}"
        )

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
