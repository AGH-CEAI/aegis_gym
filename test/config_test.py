from dataclasses import fields
from pathlib import Path

import pytest
import yaml
from clearml.utilities.proxy_object import ProxyDictPreWrite

from aegis_gym.config import ConfigManager as cm
from aegis_gym.config import LaunchArgs, parse_arguments
from aegis_gym.config.types import (
    Algorithm,
    CamerasSetup,
    CNNLayerCfg,
    Control,
    VisionEncoderCfg,
)
from aegis_gym.envs import PushTEnv


def dict_diff(actual: dict, expected: dict, path: str = ""):
    diffs = []

    all_keys = set(actual) | set(expected)

    for key in all_keys:
        current_path = f"{path}.{key}" if path else str(key)

        if key not in actual:
            diffs.append(f"Missing in actual: {current_path}")
            continue

        if key not in expected:
            diffs.append(f"Unexpected key: {current_path}")
            continue

        a = actual[key]
        e = expected[key]

        if isinstance(a, dict) and isinstance(e, dict):
            diffs.extend(dict_diff(a, e, current_path))
        elif a != e:
            diffs.append(f"{current_path}: actual={a!r}, expected={e!r}")

    return diffs


@pytest.fixture(autouse=True)
def reset_config():
    cm._global_cfg = None


def test_parse_arguments_returns_launch_args():
    result = parse_arguments(argv=[""])

    assert isinstance(result, LaunchArgs)


def test_default_arguments():
    args = parse_arguments(argv=[""])

    excluded = {
        "_args_raw",
        "control_type",
        "algorithm",
        "debug_record_dir",
        "project_name",
        "enable_recording",
    }
    for f in fields(args):
        if f.name not in excluded:
            assert getattr(args, f.name) in (None, False), (
                f"Expected '{f.name}' to be None or False by default, got {getattr(args, f.name)!r}"
            )

    assert args.control_type == Control.SIM
    assert args.algorithm == Algorithm.RL
    assert str(args.debug_record_dir).startswith("/tmp/")
    assert args.enable_recording is None


def test_record_launch_arg():
    assert parse_arguments(argv=["", "--record"]).enable_recording is True
    assert parse_arguments(argv=["", "--no-record"]).enable_recording is False


def test_get_config_before_setup_raises():
    with pytest.raises(AttributeError):
        cm.get_config()


def test_setup_config():
    cm.setup_config(argv=[""])


def test_setup_config_twice_raises():
    cm.setup_config(argv=[""])

    with pytest.raises(AttributeError):
        cm.setup_config(argv=[""])


def test_get_config():
    cm.setup_config(argv=[""])

    config = cm.get_config()
    assert config is not None


def test_env_config_image_resolution():
    cm.setup_config(argv=[""])
    val = cm.get_config().env_cfg.image_resolution

    assert isinstance(val, tuple)
    assert len(val) == 2
    assert isinstance(val[0], int)
    assert isinstance(val[1], int)


def test_env_config_cameras_setup():
    cm.setup_config(argv=[""])
    val = cm.get_config().env_cfg.cameras_setup
    default_val = cm._get_default_config_dict()["env"]["cameras_setup"]

    val_serialized = cm.get_config().env_cfg.as_dict()["cameras_setup"]

    assert isinstance(val, CamerasSetup)
    assert isinstance(val_serialized, type(default_val))
    assert val_serialized == default_val


def test_policy_bc_cfg_vision_encoder():
    cm.setup_config(argv=[""])
    val1 = cm.get_config().bc_cfg.policy.vision_encoder
    val2 = cm.get_config().bc_cfg.policy.vision_encoder_spatial

    assert isinstance(val1, VisionEncoderCfg)
    assert isinstance(val1.conv_layers, list)
    assert isinstance(val1.conv_layers[0], CNNLayerCfg)
    assert isinstance(val2, VisionEncoderCfg)
    assert isinstance(val2.conv_layers, list)
    assert isinstance(val2.conv_layers[0], CNNLayerCfg)


def test_robot_config_parsing():
    cm.setup_config(argv=[""])
    cfg = cm.get_config()

    robot_dict = cfg.robot_cfg.as_dict()
    def_dict = cm._get_default_config_dict()["robot"]

    diffs = dict_diff(robot_dict, def_dict)
    assert not diffs, "\n".join(diffs)


def test_env_specific_dict_defaults_are_resolved():
    cm.setup_config(argv=[""])
    env_dict = cm.get_config().env_cfg.env_specific_dict

    assert env_dict == cm._resolve_env_specific_dict(env_name="reacher", given=None)


def test_episode_length_defaults_to_the_selected_env():
    cm.setup_config(argv=["", "--env", "push_t"])
    env_cfg = cm.get_config().env_cfg

    assert env_cfg.episode_length_s == PushTEnv.DEFAULT_EPISODE_LENGTH_S
    assert env_cfg.max_steps == int(env_cfg.episode_length_s / env_cfg.policy_dt)


def test_action_preview_launch_arg_enables_the_debugging_tools():
    cm.setup_config(argv=["", "--debug-action-preview"])
    debug_cfg = cm.get_config().debug_cfg

    assert debug_cfg.enabled
    assert debug_cfg.enable_action_preview
    assert not debug_cfg.enable_vis_preview
    assert debug_cfg.action_preview_env_idx == 0


def test_episode_length_launch_arg_overrides_env_default():
    cm.setup_config(argv=["", "--env", "push_t", "--episode-length-s", "12.0"])

    assert cm.get_config().env_cfg.episode_length_s == 12.0


def test_push_t_preset_config_is_applied():
    preset = Path(__file__).parent.parent / "aegis_gym/config/presets/push_t.yaml"
    cm.setup_config(argv=["", "--env", "push_t", "-c", str(preset)])
    rl_cfg = cm.get_config().rl_cfg

    assert rl_cfg.algorithm.gamma == 0.995
    assert rl_cfg.policy.actor_obs_normalization is True
    assert rl_cfg.policy.noise_std_type == "log"
    _assert_robot_domain_randomization_enabled()
    goal_rand = cm.get_config().env_cfg.env_specific_dict["goal_randomization"]
    assert goal_rand["enabled"] is True
    assert goal_rand["z_rot_range_deg"] == [-180.0, 180.0]


def test_reacher_preset_config_is_applied():
    preset = Path(__file__).parent.parent / "aegis_gym/config/presets/reacher.yaml"
    cm.setup_config(argv=["", "--env", "reacher", "-c", str(preset)])
    rl_cfg = cm.get_config().rl_cfg

    assert rl_cfg.algorithm.gamma == 0.99
    assert rl_cfg.num_steps_per_env == 24
    assert rl_cfg.policy.actor_hidden_dims == [128, 128, 64]
    _assert_robot_domain_randomization_enabled()


def _assert_robot_domain_randomization_enabled() -> None:
    dr_cfg = cm.get_config().dr_cfg
    assert dr_cfg.enabled
    for rand_cfg in (
        dr_cfg.image_aug,
        dr_cfg.pd_gains,
        dr_cfg.max_speed,
        dr_cfg.cameras_extrinsics,
    ):
        assert rand_cfg.enabled, rand_cfg
    assert dr_cfg.image_aug.brightness_jitter == 0.4
    assert dr_cfg.cameras_extrinsics.tool_cams.rotation_std_deg == 0.6


def test_env_specific_dict_override_is_applied():
    cm.setup_config(argv=["", "--env-dict", '{"box_fixed": False}'])
    env_dict = cm.get_config().env_cfg.env_specific_dict

    assert env_dict["box_fixed"] is False
    assert env_dict["box_collision"] is False


def test_env_specific_dict_unknown_key_raises():
    with pytest.raises(ValueError, match="box_fixxed"):
        cm.setup_config(argv=["", "--env-dict", '{"box_fixxed": False}'])


def test_env_specific_dict_unknown_nested_key_raises():
    with pytest.raises(ValueError, match="reward_scales.keypoint"):
        cm.setup_config(argv=["", "--env-dict", '{"reward_scales": {"keypoint": 2.0}}'])


def test_env_specific_dict_unknown_key_is_ignored_when_validation_disabled():
    cm.setup_config(
        argv=["", "--env-dict", '{"box_fixxed": False}', "--ignore-env-dict-validation"]
    )
    env_dict = cm.get_config().env_cfg.env_specific_dict

    assert env_dict["box_fixxed"] is False


def test_ignore_env_dict_validation_defaults_to_false():
    args = parse_arguments(argv=[""])

    assert args.ignore_env_dict_validation is False


class _FakeSourceTask:
    def __init__(self, sections: dict):
        self._sections = sections

    def get_configuration_object_as_dict(self, name: str) -> dict | None:
        return self._sections.get(name)


def _patch_source_task(monkeypatch, sections: dict) -> None:
    monkeypatch.setattr(
        "aegis_gym.config.config_manager.Task.get_task",
        lambda task_id: _FakeSourceTask(sections),
    )


def _get_source_sections() -> dict:
    defaults = cm._get_default_config_dict()
    env = {**defaults["env"], "env_name": "push_t", "num_envs": 4096, "max_steps": 750}
    rl = defaults["rl"]
    rl["policy"]["actor_hidden_dims"] = [256, 256, 128]
    bc = {**defaults["bc"], "learning_rate": 0.123}
    return {"env": env, "rl": rl, "bc": bc}


def test_source_task_config_is_applied(monkeypatch):
    _patch_source_task(monkeypatch, _get_source_sections())
    cm.setup_config(argv=["", "--episode-length-s", "10.0"], source_task_id="task")
    cfg = cm.get_config()

    assert cfg.env_cfg.env_name == "push_t"
    assert cfg.rl_cfg.policy.actor_hidden_dims == [256, 256, 128]
    # runtime entries come from the current run
    assert cfg.env_cfg.num_envs == 10
    assert cfg.env_cfg.max_steps == int(10.0 / cfg.env_cfg.policy_dt)


def test_source_task_config_is_skipped_when_enforcing_current_config(monkeypatch):
    _patch_source_task(monkeypatch, _get_source_sections())
    cm.setup_config(argv=["", "--enforce-current-config"], source_task_id="task")

    assert cm.get_config().env_cfg.env_name == "reacher"


def test_source_task_teacher_sections_keep_the_current_bc_config(monkeypatch):
    _patch_source_task(monkeypatch, _get_source_sections())
    cm.setup_config(
        argv=["", "-a=bc"],
        source_task_id="task",
        source_sections=cm.SOURCE_TASK_TEACHER_SECTIONS,
    )
    cfg = cm.get_config()

    assert cfg.env_cfg.env_name == "push_t"
    assert cfg.rl_cfg.policy.actor_hidden_dims == [256, 256, 128]
    assert (
        cfg.bc_cfg.learning_rate == cm._get_default_config_dict()["bc"]["learning_rate"]
    )


def test_resolve_source_task_id(monkeypatch):
    class _FakeInputModel:
        def __init__(self, model_id: str):
            self.task = f"task_of_{model_id}"

    monkeypatch.setattr("aegis_gym.config.config_manager.InputModel", _FakeInputModel)

    assert cm.resolve_source_task_id(task_id="task", model_id="model") == "task"
    assert cm.resolve_source_task_id(task_id=None, model_id="model") == "task_of_model"
    assert cm.resolve_source_task_id(task_id=None, model_id=None) is None


def _write_logger_cfg_file(tmp_path, enable_policy_preview: bool):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump({"logger": {"enable_policy_preview": enable_policy_preview}})
    )
    return str(path)


def test_policy_preview_disabled_by_default():
    cm.setup_config(argv=[""])
    assert cm.get_config().logger_cfg.enable_policy_preview is False


def test_policy_preview_enabled_from_config_file(tmp_path):
    cfg_path = _write_logger_cfg_file(tmp_path, enable_policy_preview=True)
    cm.setup_config(argv=["", "-c", cfg_path])
    assert cm.get_config().logger_cfg.enable_policy_preview is True


@pytest.mark.parametrize(
    "file_value, launch_arg, expected",
    [
        (False, "--record", True),
        (True, "--no-record", False),
    ],
)
def test_policy_preview_launch_arg_overrides_config_file(
    tmp_path, file_value, launch_arg, expected
):
    cfg_path = _write_logger_cfg_file(tmp_path, enable_policy_preview=file_value)
    cm.setup_config(argv=["", "-c", cfg_path, launch_arg])
    assert cm.get_config().logger_cfg.enable_policy_preview is expected


def test_config_file_is_deep_merged_with_defaults(tmp_path):
    defaults = cm._get_default_config_dict()
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "logger": {"policy_preview_interval": 7},
                "env": {"env_specific_dict": {"box_fixed": False}},
            }
        )
    )
    cm.setup_config(argv=["", "-c", str(path)])
    cfg = cm.get_config()
    assert cfg.logger_cfg.policy_preview_interval == 7
    # The entries absent in the file keep their default values
    assert (
        cfg.logger_cfg.policy_preview_resolution
        == defaults["logger"]["policy_preview_resolution"]
    )
    assert cfg.env_cfg.num_envs == defaults["env"]["num_envs"]
    assert cfg.env_cfg.env_specific_dict["box_fixed"] is False


class _FakeAgentTask:
    """A task run by the ClearML agent: the connected config is the remote one, read-only."""

    task_id = "agent_task"

    def connect_configuration(self, configuration: dict, name: str) -> dict:
        remote = dict(configuration)
        if name == "env":
            # derived by the run the task was cloned from
            remote.update(episode_length_s=30.0, max_steps=750)
        return ProxyDictPreWrite(False, False, **remote)


def test_launch_args_are_applied_without_warnings(caplog):
    cm.setup_config(argv=["", "--num-envs", "8", "--no-record", "--env", "push_t"])

    assert cm.get_config().env_cfg.num_envs == 8
    assert "is ignored" not in caplog.text


def test_launch_args_ignored_by_read_only_config_are_reported(caplog):
    cm.setup_config(
        argv=[
            "",
            "--num-envs",
            "8",
            "--max-iterations",
            "7",
            "--env-dict",
            '{"friction": 0.9}',
        ],
        task=_FakeAgentTask(),
    )
    cfg = cm.get_config()

    # The config of the task applies, the dropped overrides are reported
    assert cfg.env_cfg.num_envs != 8
    assert "`--num-envs` (8) is ignored" in caplog.text
    assert "`--max-iterations` (7) is ignored" in caplog.text
    assert "`--env-dict` (0.9) is ignored" in caplog.text
    assert "env.env_specific_dict.friction" in caplog.text
