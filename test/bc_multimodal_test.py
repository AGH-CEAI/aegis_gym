import math

import pytest
import torch as th
import yaml
from bc_memory_test import CPU, DEFAULT_BC_CFG, _TrainingEnv, _TrainingLogger

from aegis_gym.config.types import BCCfg, PolicyBCCfg
from aegis_gym.runners.bc_policy import ExperienceBuffer, Policy, camera_modality
from aegis_gym.runners.bc_runner import BehaviorCloningRunner

pytest.importorskip("enc_mem_dec.multimodal")

STATE_OBS = ["tcp_pose", "reference_offset"]
STATE_DIMS = {"tcp_pose": 7, "reference_offset": 2}
FUSIONS = ["concat", "poe", "gpoe"]


def _multimodal(fusion: str, **kwargs) -> dict:
    return {
        "fusion": fusion,
        "feature_dim": 8,
        "latent_dim": 6,
        "state_hidden_dims": [16],
        "gate_hidden_dim": 12,
        **kwargs,
    }


def _policy_cfg(multimodal: dict, memory: dict | None = None, **policy) -> PolicyBCCfg:
    data = yaml.safe_load(DEFAULT_BC_CFG.read_text())["policy"]
    data |= {
        "action_head_state_obs_dim": None,
        "use_pose_head": False,
        "state_obs": STATE_OBS,
        "memory": memory or {"type": "none"},
        "multimodal": multimodal,
        **policy,
    }
    return PolicyBCCfg.from_dict(data)


def _policy(fusion: str, memory: dict | None = None, **multimodal) -> Policy:
    th.manual_seed(0)
    return Policy(
        cfg_policy=_policy_cfg(_multimodal(fusion, **multimodal), memory),
        action_dim=2,
        state_obs_dim=9,
        num_cameras=3,
        image_height=16,
        image_width=16,
        device=CPU,
        state_obs_dims=STATE_DIMS,
    )


def _inputs(batch: int = 4) -> tuple[th.Tensor, th.Tensor]:
    return th.rand(batch, 9, 16, 16), th.rand(batch, 9)


@pytest.mark.parametrize("fusion", FUSIONS)
def test_multimodal_policy_has_an_encoder_per_camera_and_state_input(fusion):
    policy = _policy(fusion)
    assert policy.vision_encoder is None and policy.feature_fusion is None
    names = [camera_modality(i) for i in range(3)] + STATE_OBS
    assert policy.multimodal.encoders.names == names
    rgb, state = _inputs()
    actions = policy(rgb, state)
    assert actions.shape == (4, 2)
    features = policy.last_fusion.features
    assert set(features) == set(names)
    expected_dim = 5 * 8 if fusion == "concat" else 6
    assert policy.action_head[0].in_features == expected_dim


def test_each_state_input_reaches_its_own_encoder():
    policy = _policy("concat").eval()
    rgb, state = _inputs()
    with th.no_grad():
        policy(rgb, state)
        before = {k: v.clone() for k, v in policy.last_fusion.features.items()}
        shifted = state.clone()
        shifted[:, 7:] += 1.0  # reference_offset only
        policy(rgb, shifted)
    after = policy.last_fusion.features
    assert th.equal(before["tcp_pose"], after["tcp_pose"])
    assert th.equal(before[camera_modality(0)], after[camera_modality(0)])
    assert not th.allclose(before["reference_offset"], after["reference_offset"])


def test_each_camera_reaches_its_own_encoder():
    policy = _policy("concat").eval()
    rgb, state = _inputs()
    with th.no_grad():
        policy(rgb, state)
        before = {k: v.clone() for k, v in policy.last_fusion.features.items()}
        changed = rgb.clone()
        changed[:, 3:6] = th.rand(4, 3, 16, 16)  # the second camera's channels
        policy(changed, state)
    after = policy.last_fusion.features
    assert th.equal(before[camera_modality(0)], after[camera_modality(0)])
    assert th.equal(before[camera_modality(2)], after[camera_modality(2)])
    assert not th.allclose(before[camera_modality(1)], after[camera_modality(1)])


@pytest.mark.parametrize("fusion", ["poe", "gpoe"])
def test_the_student_acts_on_the_joint_mean_and_trains_on_a_sample(fusion):
    policy = _policy(fusion)  # training mode, as in the collection
    rgb, state = _inputs()
    with th.inference_mode():
        acting = [policy(rgb, state) for _ in range(2)]
    assert th.equal(acting[0], acting[1])
    training = [policy(rgb, state) for _ in range(2)]
    assert not th.allclose(training[0], training[1])
    kl = policy.expert_kl()
    assert kl is not None and kl.requires_grad and kl.item() > 0


@pytest.mark.parametrize("fusion", ["poe", "gpoe"])
def test_fusion_stats_share_the_joint_between_the_modalities(fusion):
    policy = _policy(fusion, prior_expert=True)
    policy(*_inputs())
    stats = policy.fusion_stats()
    names = [camera_modality(i) for i in range(3)] + STATE_OBS + ["prior"]
    assert math.isclose(sum(stats[f"share_{n}"] for n in names), 1.0, rel_tol=1e-5)
    if fusion == "gpoe":
        assert math.isclose(sum(stats[f"weight_{n}"] for n in names), 1.0, rel_tol=1e-5)
        # the gate starts with equal weights
        assert all(
            math.isclose(stats[f"weight_{n}"], 1 / len(names), rel_tol=1e-5)
            for n in names
        )
    else:
        assert not any(k.startswith("weight_") for k in stats)


def test_concat_has_no_experts():
    policy = _policy("concat")
    policy(*_inputs())
    assert policy.expert_kl() is None and policy.fusion_stats() == {}


def test_multimodal_with_a_memory_on_chunks():
    memory = {
        "type": "gru",
        "hidden_dim": 8,
        "residual": True,
        "sequence_length": 3,
        "burn_in": 2,
    }
    policy = _policy("gpoe", memory)
    assert policy.memory.input_dim == 6
    rgb = th.rand(5, 2, 9, 16, 16)
    state = th.rand(5, 2, 9)
    starts = th.zeros(5, 2, dtype=th.bool)
    starts[0] = True
    actions = policy.sequence(rgb, state, starts, burn_in=2)
    assert actions.shape == (3, 2, 2)
    # the KL of the trained steps (after the burn-in), with gradients
    assert policy.expert_kl().requires_grad


def test_multimodal_config_errors():
    cfg = _policy_cfg(_multimodal("gpoe"))
    kwargs = {
        "action_dim": 2,
        "num_cameras": 3,
        "image_height": 16,
        "image_width": 16,
        "device": CPU,
    }
    with pytest.raises(ValueError, match="state_obs_dims"):
        Policy(cfg_policy=cfg, state_obs_dim=9, **kwargs)
    with pytest.raises(ValueError, match="don't match"):
        Policy(
            cfg_policy=cfg, state_obs_dim=9, state_obs_dims={"tcp_pose": 9}, **kwargs
        )
    with pytest.raises(ValueError, match="use_pose_head"):
        Policy(
            cfg_policy=_policy_cfg(_multimodal("gpoe"), use_pose_head=True),
            state_obs_dim=9,
            state_obs_dims=STATE_DIMS,
            **kwargs,
        )
    # one state input: its dimension is the state_obs_dim
    single = _policy_cfg(_multimodal("poe"), state_obs=["tcp_pose"])
    policy = Policy(cfg_policy=single, state_obs_dim=7, **kwargs)
    assert policy.multimodal.encoders.names[-1] == "tcp_pose"


def _training_runner(
    tmp_path, fusion: str, kl_weight: float, memory: dict
) -> BehaviorCloningRunner:
    data = yaml.safe_load(DEFAULT_BC_CFG.read_text())
    data |= {
        "num_steps_per_env": 6,
        "buffer_size": 20,
        "num_epochs": 2,
        "num_mini_batches": 2,
        "save_freq": 1000,
        "use_teacher_mixing": True,
        "image_buffer_dtype": "uint8",
    }
    data["policy"] |= {
        "use_pose_head": False,
        "action_head_state_obs_dim": None,
        "state_obs": STATE_OBS,
        "memory": memory,
        "multimodal": _multimodal(fusion, kl_weight=kl_weight),
    }
    cfg = BCCfg.from_dict(data)
    env = _TrainingEnv()
    runner = object.__new__(BehaviorCloningRunner)
    runner.env, runner.device, runner.cfg_train = env, CPU, cfg
    runner._log_dir = tmp_path
    runner._teacher = lambda obs: th.rand(env.num_envs, 2) * 2 - 1
    runner._extract_config()
    runner.logger = _TrainingLogger()
    runner._current_iter, runner._best_model_iter = 0, -1
    runner._best_model_score = (-math.inf, -math.inf)
    th.manual_seed(0)
    runner._policy = Policy(
        cfg_policy=cfg.policy,
        action_dim=2,
        state_obs_dim=9,
        num_cameras=3,
        image_height=16,
        image_width=16,
        device=CPU,
        state_obs_dims=STATE_DIMS,
    )
    runner._optimizer = th.optim.Adam(runner._policy.parameters(), lr=1e-3)
    runner._buffer = ExperienceBuffer(
        num_envs=env.num_envs,
        max_size=cfg.buffer_size,
        img_shape=(9, 16, 16),
        state_dim=9,
        action_dim=2,
        dtype=th.float32,
        image_dtype=th.uint8,
    )
    return runner


@pytest.mark.parametrize(
    "memory",
    [{"type": "none"}, {"type": "identity", "sequence_length": 4, "burn_in": 4}],
    ids=["none", "identity"],
)
@pytest.mark.parametrize(
    "fusion,kl_weight", [("concat", 0.0), ("poe", 0.0), ("gpoe", 0.01)]
)
def test_training_iterations_update_the_multimodal_student(
    tmp_path, fusion, kl_weight, memory
):
    runner = _training_runner(tmp_path, fusion, kl_weight, memory)
    before = {k: v.clone() for k, v in runner._policy.state_dict().items()}
    runner.learn(num_learning_iterations=5)

    scalars = runner.logger.scalars
    assert len(scalars["Loss/action"]) == 5
    assert all(math.isfinite(v) for v in scalars["Loss/total"])
    after = runner._policy.state_dict()
    changed = {k for k in before if not th.equal(before[k], after[k])}
    for name in runner._policy.multimodal.encoders.names:
        assert any(
            k.startswith(f"multimodal.encoders.encoders.{name}.") for k in changed
        ), name
    if fusion == "concat":
        assert "Loss/expert_kl" not in scalars and "BC/expert_kl" not in scalars
        return
    assert any(k.startswith("multimodal.heads.") for k in changed)
    assert all(math.isfinite(v) and v > 0 for v in scalars["BC/expert_kl"])
    expected_kl_loss = [kl_weight * v for v in scalars["BC/expert_kl"]]
    assert scalars["Loss/expert_kl"] == pytest.approx(expected_kl_loss)
    assert len(scalars["BC/fusion_share_camera_0"]) == 5
    if fusion == "gpoe":
        assert any(k.startswith("multimodal.gate.") for k in changed)
        assert len(scalars["BC/fusion_weight_reference_offset"]) == 5


def test_modality_dropout_leaves_modalities_out_in_training_only():
    policy = _policy("gpoe", modality_dropout=0.5).train()
    rgb, state = _inputs(256)

    policy(rgb, state)
    weights = policy.last_fusion.weights  # [B, experts, latent]
    left_out = (weights == 0).all(dim=-1)
    with th.no_grad():
        policy.eval()(rgb, state)

    assert 0.3 < left_out.float().mean() < 0.6
    assert (~left_out).any(dim=1).all()  # at least one modality stays
    assert not (policy.last_fusion.weights == 0).any()


def test_modality_dropout_only_of_the_named_modalities():
    policy = _policy(
        "gpoe", modality_dropout=1.0, modality_dropout_names=["camera_0"]
    ).train()
    rgb, state = _inputs(32)

    policy(rgb, state)
    left_out = (policy.last_fusion.weights == 0).all(dim=-1)

    assert left_out[:, 0].all() and not left_out[:, 1:].any()


def test_modality_dropout_is_the_same_on_all_the_steps_of_a_chunk():
    policy = _policy("gpoe", modality_dropout=0.5).train()
    rgb, state = th.rand(5, 32, 9, 16, 16), th.rand(5, 32, 9)

    policy._chunk_features(rgb, state)
    left_out = (policy.last_fusion.weights == 0).all(dim=-1).view(5, 32, -1)

    assert (left_out == left_out[:1]).all()
