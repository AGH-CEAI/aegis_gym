from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from clearml.binding.frameworks import WeightsFileHandler

from aegis_gym.aux.utils import clearml_models
from aegis_gym.aux.utils.clearml_models import (
    checkpoint_input_model,
    connect_checkpoint_input_model,
    import_checkpoint_artifact,
    link_teacher_task,
    without_clearml_load_binding,
)

_MODULE = "aegis_gym.aux.utils.clearml_models"


def _source_task() -> MagicMock:
    task = MagicMock(id="rl_task_id", project="rl_project_id")
    task.name = "PUSH_T_TRAIN_rl"
    task.artifacts = {
        "model_100": SimpleNamespace(url="https://files/rl/model_100.pt"),
        "model_1000": SimpleNamespace(url="https://files/rl/model_1000.pt"),
        "model_best": SimpleNamespace(url="https://files/rl/model_best.pt"),
    }
    task.get_output_log_web_page.return_value = "https://clearml/rl_task_id/output/log"
    return task


@pytest.fixture
def import_model(monkeypatch) -> MagicMock:
    """`InputModel.import_model()`, returning a draft model of an earlier anonymous import."""
    model = MagicMock(published=False, tags=["earlier"])
    model.name = "PUSH_T_TRAIN_rl / model_1000"
    import_model = MagicMock(return_value=model)
    monkeypatch.setattr(f"{_MODULE}.InputModel.import_model", import_model)
    return import_model


def test_checkpoint_artifact_is_imported_linked_to_its_task(import_model):
    model = import_checkpoint_artifact(_source_task(), "model_1000")

    import_model.assert_called_once_with(
        weights_url="https://files/rl/model_1000.pt", framework="PyTorch"
    )
    update = model._get_base_model().update.call_args.kwargs
    assert update["name"] == "PUSH_T_TRAIN_rl / model_1000"
    assert (update["task_id"], update["project_id"]) == ("rl_task_id", "rl_project_id")
    assert "https://clearml/rl_task_id/output/log" in update["comment"]
    assert update["tags"] == ["earlier", "checkpoint", "source_task:rl_task_id"]
    metadata = model.set_all_metadata.call_args.args[0]
    assert {k: v["value"] for k, v in metadata.items()} == {
        "source_task_id": "rl_task_id",
        "source_task_name": "PUSH_T_TRAIN_rl",
        "artifact": "model_1000",
        "iteration": "1000",
    }


def test_published_checkpoint_model_is_not_edited(import_model):
    import_model.return_value.published = True

    model = import_checkpoint_artifact(_source_task(), "model_1000")

    model._get_base_model().update.assert_not_called()
    model.set_all_metadata.assert_not_called()


def test_latest_checkpoint_of_the_task_is_connected(monkeypatch, import_model):
    current_task = MagicMock()
    monkeypatch.setattr(f"{_MODULE}.Task.current_task", lambda: current_task)
    monkeypatch.setattr(f"{_MODULE}.Task.get_task", lambda task_id: _source_task())

    model = connect_checkpoint_input_model(
        name="rl_teacher", clearml_task_id="rl_task_id"
    )

    assert import_model.call_args.kwargs["weights_url"].endswith("model_1000.pt")
    model.connect.assert_called_once_with(
        task=current_task, name="rl_teacher", ignore_remote_overrides=True
    )


def test_given_model_is_connected_as_is(monkeypatch):
    current_task = MagicMock()
    monkeypatch.setattr(f"{_MODULE}.Task.current_task", lambda: current_task)
    input_model = MagicMock()
    monkeypatch.setattr(f"{_MODULE}.InputModel", input_model)

    # the model takes precedence over the task, as in `resolve_checkpoint()`
    connect_checkpoint_input_model(
        name="rl_teacher", clearml_task_id="rl_task_id", clearml_model_id="model_id"
    )

    input_model.assert_called_once_with(model_id="model_id")
    input_model.return_value.connect.assert_called_once()


def test_nothing_is_connected_without_a_clearml_checkpoint_or_task(monkeypatch):
    monkeypatch.setattr(f"{_MODULE}.Task.current_task", lambda: MagicMock())
    assert connect_checkpoint_input_model(name="rl_teacher") is None

    monkeypatch.setattr(f"{_MODULE}.Task.current_task", lambda: None)
    assert (
        connect_checkpoint_input_model(name="rl_teacher", clearml_task_id="x") is None
    )


def test_connection_failure_is_only_logged(monkeypatch):
    monkeypatch.setattr(f"{_MODULE}.Task.current_task", lambda: MagicMock())
    monkeypatch.setattr(
        f"{_MODULE}.InputModel", MagicMock(side_effect=ValueError("no such model"))
    )

    assert connect_checkpoint_input_model(name="x", clearml_model_id="bad") is None


def _restore_weights_file(path: str) -> None:
    """ClearML's `torch.load()` binding, with the model registration mocked out."""
    WeightsFileHandler.restore_weights_file(
        model=None, filepath=path, framework="PyTorch", task=MagicMock()
    )


def test_load_binding_is_suspended_within_the_context(monkeypatch, tmp_path):
    binding_import = MagicMock()
    monkeypatch.setattr(
        "clearml.binding.frameworks.InputModel.import_model", binding_import
    )
    monkeypatch.setattr(
        f"{_MODULE}.connect_checkpoint_input_model", lambda **kwargs: MagicMock()
    )
    callbacks = dict(WeightsFileHandler._model_pre_callbacks)

    with checkpoint_input_model(name="rl_teacher", clearml_task_id="rl_task_id"):
        _restore_weights_file(str(tmp_path / "model_1000.pt"))
    binding_import.assert_not_called()
    assert WeightsFileHandler._model_pre_callbacks == callbacks

    with without_clearml_load_binding():
        _restore_weights_file(str(tmp_path / "model_1000.pt"))
    binding_import.assert_not_called()


def test_load_binding_is_kept_if_nothing_was_connected(monkeypatch):
    monkeypatch.setattr(
        f"{_MODULE}.connect_checkpoint_input_model", lambda **kwargs: None
    )
    callbacks = dict(WeightsFileHandler._model_pre_callbacks)

    with checkpoint_input_model(name="rl_teacher") as model:
        assert model is None
        assert WeightsFileHandler._model_pre_callbacks == callbacks


def test_save_binding_is_kept():
    info = MagicMock()
    save = WeightsFileHandler.CallbackType.save
    assert clearml_models._skip_load_binding(save, info) is info


def test_bc_task_is_linked_to_its_teacher_task():
    task = MagicMock()

    link_teacher_task(task=task, teacher_task_id="rl_task_id")

    task.set_parent.assert_called_once_with("rl_task_id")
    task.add_tags.assert_called_once_with(["rl_teacher:task/rl_task_id"])
