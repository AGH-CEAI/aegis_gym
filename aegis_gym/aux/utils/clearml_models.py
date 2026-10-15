import re
from collections.abc import Iterator
from contextlib import contextmanager

from clearml import InputModel, Task
from clearml.binding.frameworks import WeightsFileHandler

from aegis_gym.aux.logging import get_logger

_LOGGER_NAME = "ClearmlModels"
CHECKPOINT_MODEL_TAG = "checkpoint"


def list_clearml_checkpoint_artifacts(
    task: Task, clearml_artifact_name: str = "model"
) -> list[tuple[int, str]]:
    """The (iteration, artifact name) of the task's `<clearml_artifact_name>_<N>` checkpoints, by
    iteration (e.g. without the `model_best`)."""
    pattern = re.compile(rf"^{re.escape(clearml_artifact_name)}_(\d+)$")
    matched = [
        (int(m.group(1)), name)
        for name in task.artifacts
        if (m := pattern.match(name)) is not None
    ]
    return sorted(matched)


@contextmanager
def checkpoint_input_model(
    name: str,
    clearml_task_id: str | None = None,
    clearml_model_id: str | None = None,
    clearml_artifact_name: str = "model",
) -> Iterator[InputModel | None]:
    """
    Connects the ClearML checkpoint loaded within the context to the current task as its input
    model `name`, see `connect_checkpoint_input_model()`. ClearML's `torch.load()` binding is
    suspended meanwhile, otherwise it would connect the loaded file once more, as an anonymous
    "Imported by <task>" model. Without the ClearML checkpoint (a local one) the binding is kept.
    """
    model = connect_checkpoint_input_model(
        name=name,
        clearml_task_id=clearml_task_id,
        clearml_model_id=clearml_model_id,
        clearml_artifact_name=clearml_artifact_name,
    )
    if model is None:
        yield None
        return
    with without_clearml_load_binding():
        yield model


@contextmanager
def without_clearml_load_binding() -> Iterator[None]:
    """Suspends ClearML's `torch.load()` binding, which connects every loaded file to the current
    task as an input model (an anonymous "Imported by <task>" one, if the file isn't a model's)."""
    handle = WeightsFileHandler.add_pre_callback(_skip_load_binding)
    try:
        yield
    finally:
        WeightsFileHandler.remove_pre_callback(handle)


def connect_checkpoint_input_model(
    name: str,
    clearml_task_id: str | None = None,
    clearml_model_id: str | None = None,
    clearml_artifact_name: str = "model",
) -> InputModel | None:
    """
    Connects the checkpoint resolved as in `resolve_checkpoint()` to the current task as its input
    model `name`: the ClearML model `clearml_model_id` as is, otherwise the latest
    `<clearml_artifact_name>_<N>` artifact of the task `clearml_task_id`, as a model linked to that
    task (see `import_checkpoint_artifact()`). Failures are only logged, the training doesn't
    depend on them.
    """
    task = Task.current_task()
    if task is None or (clearml_task_id is None and clearml_model_id is None):
        return None
    logger = get_logger(_LOGGER_NAME)
    try:
        if clearml_model_id is not None:
            model = InputModel(model_id=clearml_model_id)
        else:
            source_task = Task.get_task(task_id=clearml_task_id)
            checkpoints = list_clearml_checkpoint_artifacts(
                source_task, clearml_artifact_name
            )
            if not checkpoints:
                return None
            _, artifact_name = checkpoints[-1]
            model = import_checkpoint_artifact(source_task, artifact_name)
        # it's the model loaded from the launch args, also under the ClearML agent
        model.connect(task=task, name=name, ignore_remote_overrides=True)
    except Exception as err:  # noqa: BLE001 (the SDK and the server errors alike)
        logger.warning(f"Failed to connect the input model '{name}': {err}")
        return None
    logger.info(f"Connected the input model '{name}': {model.name} ({model.id})")
    return model


def import_checkpoint_artifact(source_task: Task, artifact_name: str) -> InputModel:
    """
    The ClearML model of the checkpoint artifact `artifact_name` of the `source_task`, named and
    described after it, in its project and linked to it as the task that created the model (with
    the `source_task_id`, `source_task_name`, `artifact` and `iteration` metadata).

    The model of the artifact's URL is reused if it exists, e.g. imported by an earlier run as an
    anonymous "Imported by <task>" model, and described the same way (unless it's published).
    """
    artifact = source_task.artifacts[artifact_name]
    model = InputModel.import_model(weights_url=artifact.url, framework="PyTorch")
    if model.published:
        return model

    iteration = artifact_name.rsplit("_", maxsplit=1)[-1]
    tags = [CHECKPOINT_MODEL_TAG, f"source_task:{source_task.id}"]
    # the backend model, the public setters update the fields one request each and miss the task
    model._get_base_model().update(
        name=f"{source_task.name} / {artifact_name}",
        comment=(
            f"Checkpoint `{artifact_name}` (artifact) of the ClearML task '{source_task.name}' "
            f"({source_task.id}):\n{source_task.get_output_log_web_page()}"
        ),
        tags=list(dict.fromkeys([*(model.tags or []), *tags])),
        task_id=source_task.id,
        project_id=source_task.project,
    )
    model.set_all_metadata(
        {
            "source_task_id": {"value": source_task.id, "type": "str"},
            "source_task_name": {"value": source_task.name, "type": "str"},
            "artifact": {"value": artifact_name, "type": "str"},
            "iteration": {"value": iteration, "type": "int"},
        },
        replace=False,
    )
    return model


def _skip_load_binding(
    callback_type: WeightsFileHandler.CallbackType,
    model_info: WeightsFileHandler.ModelInfo,
) -> WeightsFileHandler.ModelInfo | None:
    """`WeightsFileHandler` pre-callback skipping the model registration of the loaded files."""
    if callback_type == WeightsFileHandler.CallbackType.load:
        return None
    return model_info


def link_teacher_task(task: Task, teacher_task_id: str) -> None:
    """
    Links the BC `task` to the ClearML task of its RL teacher: as its parent (the lineage in the
    ClearML UI) and with the `rl_teacher:task/<ID>` tag, which is kept also when a resume of the BC
    replaces the parent with the resumed task.
    """
    task.set_parent(teacher_task_id)
    task.add_tags([f"rl_teacher:task/{teacher_task_id}"])
