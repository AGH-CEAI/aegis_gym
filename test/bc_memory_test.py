import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th
import yaml

import aegis_gym.config
from aegis_gym.config.types import MemoryCfg, PolicyBCCfg
from aegis_gym.runners.bc_policy import (
    ExperienceBuffer,
    Policy,
    build_memory,
    latent_pose_errors,
    latent_pose_target,
)
from aegis_gym.runners.bc_runner import BehaviorCloningRunner
from aegis_gym.runners.policy_preview import make_visual_bc_policy

NUM_ENVS = 3
CPU = th.device("cpu")
DEFAULT_BC_CFG = Path(aegis_gym.config.__file__).parent / "defaults" / "bc.yaml"


def _buffer(max_size: int, num_steps: int, starts_every: int = 4) -> ExperienceBuffer:
    """Step `t` of env `e` has the action label `(t, e)`; env `e` starts episodes every few steps."""
    buffer = ExperienceBuffer(
        num_envs=NUM_ENVS,
        max_size=max_size,
        img_shape=(3, 2, 2),
        state_dim=1,
        action_dim=2,
        dtype=th.float32,
        image_dtype=th.uint8,
    )
    for t in range(num_steps):
        envs = th.arange(NUM_ENVS, dtype=th.float32)
        buffer.add(
            rgb_obs=th.zeros(NUM_ENVS, 3, 2, 2),
            robot_pose=th.full((NUM_ENVS, 1), float(t)),
            object_poses=th.zeros(NUM_ENVS, 7),
            actions=th.stack([th.full((NUM_ENVS,), float(t)), envs], dim=-1),
            episode_starts=(t + th.arange(NUM_ENVS)) % starts_every == 0,
        )
    return buffer


def _sequence_batches(buffer: ExperienceBuffer, **kwargs) -> list[dict]:
    args = {"num_mini_batches": 2, "num_epochs": 1, "sequence_length": 3, "burn_in": 2}
    return list(buffer.get_sequence_batches(**(args | kwargs)))


def test_sequence_chunks_are_consecutive_steps_of_one_env_within_the_ring():
    # 13 steps into a ring of 10: the steps 3..12 are kept, the write position is inside the ring
    batches = _sequence_batches(_buffer(max_size=10, num_steps=13))
    trained = []
    for batch in batches:
        assert batch["burn_in"] == 2
        steps, envs = batch["actions"][..., 0], batch["actions"][..., 1]
        assert batch["actions"].shape[0] == 5
        assert th.all(steps[1:] - steps[:-1] == 1)
        assert th.all(envs == envs[0])
        assert steps.min() >= 3 and steps.max() <= 12
        assert th.equal(batch["robot_pose"][..., 0], steps)
        expected_starts = (steps.long() + envs.long()) % 4 == 0
        assert th.equal(batch["episode_starts"], expected_starts)
        trained += [
            (int(t), int(e)) for t, e in zip(steps[2:].flatten(), envs[2:].flatten())
        ]
    # the trained parts don't overlap
    assert len(trained) == len(set(trained))


def test_sequence_chunks_shrink_while_the_buffer_fills():
    batches = _sequence_batches(_buffer(max_size=10, num_steps=4), sequence_length=3)
    assert {b["burn_in"] for b in batches} == {1}
    assert all(b["actions"].shape[0] == 4 for b in batches)

    batches = _sequence_batches(_buffer(max_size=10, num_steps=2), sequence_length=3)
    assert {b["burn_in"] for b in batches} == {0}
    assert all(b["actions"].shape[0] == 2 for b in batches)


def test_sequence_mini_batches_cover_all_the_chunks_of_an_epoch():
    batches = _sequence_batches(
        _buffer(max_size=20, num_steps=20), num_mini_batches=3, burn_in=0
    )
    assert sum(b["actions"].shape[1] for b in batches) == 20 // 3 * NUM_ENVS
    assert len(batches) == 3


def _policy_cfg(latent_pose_loss_weight: float = 0.0, **memory) -> PolicyBCCfg:
    data = yaml.safe_load(DEFAULT_BC_CFG.read_text())["policy"]
    data["action_head_state_obs_dim"] = None
    data["use_pose_head"] = False
    data["memory"] = memory
    data["latent_pose_loss_weight"] = latent_pose_loss_weight
    return PolicyBCCfg.from_dict(data)


def _policy(latent_pose_loss_weight: float = 0.0, **memory) -> Policy:
    th.manual_seed(0)
    policy = Policy(
        cfg_policy=_policy_cfg(latent_pose_loss_weight, **memory),
        action_dim=2,
        state_obs_dim=4,
        num_cameras=3,
        image_height=16,
        image_width=16,
        device=CPU,
    )
    return (
        policy.eval()
    )  # BatchNorm with the running statistics: the same per step and chunk


def test_memoryless_policy_needs_no_enc_mem_dec():
    policy = _policy(type="none")
    assert not policy.has_memory
    assert policy(th.rand(NUM_ENVS, 9, 16, 16), th.rand(NUM_ENVS, 4)).shape == (
        NUM_ENVS,
        2,
    )
    with pytest.raises(ValueError, match="residual"):
        build_memory(MemoryCfg(type="none", residual=True), input_dim=4)


@pytest.mark.parametrize(
    "memory",
    [
        {"type": "gru", "hidden_dim": 8},
        {"type": "lstm", "hidden_dim": 8, "residual": True},
        {"type": "frame_stack", "num_frames": 3},
        {"type": "frame_stack", "num_frames": 3, "frame_stride": 2},
        {"type": "attention_window", "window": 3, "model_dim": 8, "num_heads": 2},
    ],
    ids=lambda m: (
        m["type"]
        + ("+residual" if m.get("residual") else "")
        + (f"_stride{m['frame_stride']}" if m.get("frame_stride") else "")
    ),
)
def test_policy_chunk_actions_equal_the_steps_after_the_burn_in(memory):
    pytest.importorskip("enc_mem_dec")
    policy = _policy(**memory)
    assert policy.has_memory
    with pytest.raises(RuntimeError, match="memory"):
        policy(th.rand(NUM_ENVS, 9, 16, 16), th.rand(NUM_ENVS, 4))

    num_steps, burn_in = 6, 2
    rgb = th.rand(num_steps, NUM_ENVS, 9, 16, 16)
    state = th.rand(num_steps, NUM_ENVS, 4)
    starts = th.zeros(num_steps, NUM_ENVS, dtype=th.bool)
    starts[1, 0] = starts[3, 1] = starts[4, 2] = True

    with th.no_grad():
        memory_state = policy.initial_memory_state(NUM_ENVS)
        stepped = []
        for t in range(num_steps):
            action, memory_state = policy.step(
                rgb[t], state[t], memory_state, starts[t]
            )
            stepped.append(action)
        chunk = policy.sequence(rgb, state, starts, burn_in=burn_in)

    assert chunk.shape == (num_steps - burn_in, NUM_ENVS, 2)
    assert th.allclose(chunk, th.stack(stepped)[burn_in:], atol=1e-5)


class _RecordingMemoryStudent:
    """A student with a memory counting its steps per env: action = steps since the reset."""

    has_memory = True

    def __init__(self):
        self.starts: list[list[bool]] = []

    def initial_memory_state(self, batch_size: int):
        return (th.zeros(batch_size),)

    def step(self, rgb, state, memory_state, episode_starts):
        self.starts.append(episode_starts.tolist())
        count = th.where(episode_starts, 0.0, memory_state[0]) + 1
        return count.unsqueeze(-1).repeat(1, 2), (count,)


class _CountingEnv:
    num_envs = 2

    def __init__(self):
        self.episode_length_buf = th.zeros(self.num_envs, dtype=th.long)
        tcp_pose = th.zeros(self.num_envs, 7)
        self.manipulator = SimpleNamespace(get_tcp_pose=lambda: tcp_pose)
        self.object = SimpleNamespace(get_pose=lambda: th.zeros(self.num_envs, 7))
        self.unwrapped = self

    def get_modality_observations(self, modalities) -> dict:
        return {m: th.zeros(self.num_envs, 3, 2, 2) for m in modalities}


def test_inference_policy_carries_the_memory_and_resets_it_at_episode_starts():
    env, student = _CountingEnv(), _RecordingMemoryStudent()
    policy = make_visual_bc_policy(env=env, bc_policy=student, state_obs=["tcp_pose"])

    counts = []
    for t in range(4):
        counts.append(policy(None)[:, 0].tolist())
        env.episode_length_buf += 1
        if t == 1:
            env.episode_length_buf[1] = 0  # env 1 reset after its 2nd step

    assert student.starts[0] == [True, True]
    assert counts == [[1, 1], [2, 2], [3, 1], [4, 2]]


def test_collection_carries_the_student_memory_and_stores_the_episode_starts():
    env, student = _CountingEnv(), _RecordingMemoryStudent()
    stored = []
    runner = object.__new__(BehaviorCloningRunner)
    runner.env = env
    runner.logger = SimpleNamespace(process_env_step=lambda **kwargs: None)
    runner._num_steps_per_env = 3
    runner._use_teacher_mixing = False
    runner._clip_teacher_actions = False
    runner._state_obs = ["tcp_pose"]
    runner._teacher = lambda obs: th.zeros(env.num_envs, 2)
    runner._policy = student
    runner._memory_state = None
    runner._buffer = SimpleNamespace(add=lambda *args, **kwargs: stored.append(kwargs))
    executed = []

    def step(actions):
        executed.append(actions[:, 0].tolist())
        env.episode_length_buf += 1
        env.episode_length_buf[0] = 0  # env 0 resets after every step
        return None, th.zeros(env.num_envs), th.zeros(env.num_envs), {}

    env.get_observations = lambda: None
    env.step = step
    runner._collect_with_rl_teacher()
    runner._collect_with_rl_teacher()  # the next iteration continues the episodes

    assert executed == [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6]]
    assert stored[0]["episode_starts"].tolist() == [True, True]
    assert stored[1]["episode_starts"].tolist() == [True, False]


def _tee_poses(xy_yaw: list[tuple[float, float, float]]) -> th.Tensor:
    """[N, 7] (x, y, z, qw, qx, qy, qz) tee poses with the yaw about z."""
    rows = []
    for x, y, yaw in xy_yaw:
        rows.append([x, y, 0.02, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    return th.tensor(rows)


def test_latent_pose_target_is_planar_in_dm():
    target = latent_pose_target(_tee_poses([(0.47, -0.05, math.pi / 2)]))
    assert target.tolist()[0] == pytest.approx([4.7, -0.5, 1.0, 0.0], abs=1e-5)


def test_latent_pose_errors_in_m_and_rad_with_the_yaw_wrapped():
    poses = _tee_poses([(0.4, 0.0, math.radians(179)), (0.4, 0.0, 0.0)])
    pred = latent_pose_target(
        _tee_poses([(0.4, 0.1, math.radians(-179)), (0.43, 0.04, 0.0)])
    )
    pos_err, yaw_err = latent_pose_errors(pred, poses)
    assert pos_err.tolist() == pytest.approx([0.1, 0.05], abs=1e-5)
    assert yaw_err.tolist() == pytest.approx([math.radians(2), 0.0], abs=1e-5)


def test_latent_pose_decoder_reads_the_action_head_input():
    rgb, state = th.rand(NUM_ENVS, 9, 16, 16), th.rand(NUM_ENVS, 4)
    assert _policy().latent_pose_head is None
    assert _policy().predict_latent_pose(th.rand(NUM_ENVS, 8)) is None

    policy = _policy(latent_pose_loss_weight=0.5)
    _, latent = policy.forward_with_latent(rgb, state)
    assert policy.predict_latent_pose(latent).shape == (NUM_ENVS, 4)


def test_latent_pose_decoder_on_the_memory_output():
    pytest.importorskip("enc_mem_dec")
    policy = _policy(
        latent_pose_loss_weight=0.5, type="gru", hidden_dim=8, residual=True
    )
    num_steps = 5
    starts = th.zeros(num_steps, NUM_ENVS, dtype=th.bool)
    actions, latents = policy.sequence(
        th.rand(num_steps, NUM_ENVS, 9, 16, 16),
        th.rand(num_steps, NUM_ENVS, 4),
        starts,
        burn_in=2,
        return_latents=True,
    )
    assert actions.shape == (3, NUM_ENVS, 2)
    assert latents.shape == (3, NUM_ENVS, policy.memory.output_dim)
    assert policy.predict_latent_pose(latents).shape == (3, NUM_ENVS, 4)


class _TrainingEnv:
    """3 envs with images, the tee pose and episodes of 7 steps (env e starts e steps later)."""

    num_envs = 3
    device = CPU

    def __init__(self):
        self.episode_length_buf = th.arange(self.num_envs)
        self.unwrapped = self
        self.manipulator = SimpleNamespace(
            get_tcp_pose=lambda: th.rand(self.num_envs, 7)
        )
        self.object = SimpleNamespace(get_pose=self._tee_poses)

    def _tee_poses(self) -> th.Tensor:
        yaw = th.rand(self.num_envs) * 2 * math.pi
        xy = th.rand(self.num_envs, 2) * 0.2
        quat = th.stack(
            [th.cos(yaw / 2), th.zeros_like(yaw), th.zeros_like(yaw), th.sin(yaw / 2)],
            -1,
        )
        return th.cat([xy, th.full((self.num_envs, 1), 0.02), quat], dim=-1)

    def get_observations(self) -> th.Tensor:
        return th.rand(self.num_envs, 12)

    def get_modality_observations(self, modalities) -> dict:
        return {m: th.rand(self.num_envs, 3, 16, 16) for m in modalities}

    def step(self, actions: th.Tensor):
        self.episode_length_buf += 1
        dones = self.episode_length_buf >= 7
        self.episode_length_buf[dones] = 0
        return self.get_observations(), th.zeros(self.num_envs), dones, {}


class _TrainingLogger:
    def __init__(self):
        self.scalars: dict[str, list[float]] = {}
        self.writer = SimpleNamespace(add_scalar=self._add)
        self.rewbuffer, self.lenbuffer, self.ep_extras = [], [], []

    def _add(self, tag: str, value: float, it: int) -> None:
        self.scalars.setdefault(tag, []).append(float(value))

    def process_env_step(self, **kwargs) -> None: ...

    def save_model(self, **kwargs) -> None: ...


def _training_runner(tmp_path, latent_pose_loss_weight: float, memory: dict):
    from aegis_gym.config.types import BCCfg

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
        "memory": memory,
        "latent_pose_loss_weight": latent_pose_loss_weight,
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
        state_obs_dim=7,
        num_cameras=3,
        image_height=16,
        image_width=16,
        device=CPU,
    )
    runner._optimizer = th.optim.Adam(runner._policy.parameters(), lr=1e-3)
    runner._buffer = ExperienceBuffer(
        num_envs=env.num_envs,
        max_size=cfg.buffer_size,
        img_shape=(9, 16, 16),
        state_dim=7,
        action_dim=2,
        dtype=th.float32,
        image_dtype=th.uint8,
    )
    return runner


@pytest.mark.parametrize("latent_pose_loss_weight", [0.0, 1.0], ids=["", "latent_pose"])
@pytest.mark.parametrize(
    "memory",
    [
        {"type": "none"},
        {"type": "identity", "sequence_length": 4, "burn_in": 4},
        {
            "type": "gru",
            "hidden_dim": 8,
            "residual": True,
            "sequence_length": 4,
            "burn_in": 4,
        },
        {"type": "lstm", "hidden_dim": 8, "sequence_length": 4, "burn_in": 2},
        {"type": "frame_stack", "num_frames": 3, "sequence_length": 4, "burn_in": 2},
        {
            "type": "attention_window",
            "window": 5,
            "model_dim": 8,
            "num_heads": 2,
            "residual": True,
            "sequence_length": 4,
            "burn_in": 4,
        },
    ],
    ids=lambda m: m["type"],
)
def test_training_iterations_update_the_student(
    tmp_path, memory, latent_pose_loss_weight
):
    if memory["type"] != "none":
        pytest.importorskip("enc_mem_dec")
    runner = _training_runner(tmp_path, latent_pose_loss_weight, memory)
    before = {k: v.clone() for k, v in runner._policy.state_dict().items()}

    # the buffer fills in 4 iterations of 6 steps: the chunks and the burn-in shrink at first
    runner.learn(num_learning_iterations=5)

    scalars = runner.logger.scalars
    assert len(scalars["Loss/action"]) == 5
    assert all(math.isfinite(v) for v in scalars["Loss/total"])
    after = runner._policy.state_dict()
    changed = {k for k in before if not th.equal(before[k], after[k])}
    assert any(k.startswith("action_head") for k in changed)
    if runner._policy.memory is not None and any(
        True for _ in runner._policy.memory.parameters()
    ):
        assert any(k.startswith("memory.") for k in changed)
    if latent_pose_loss_weight > 0:
        assert any(k.startswith("latent_pose_head") for k in changed)
        assert len(scalars["BC/latent_pose_pos_err_cm"]) == 5
        assert all(math.isfinite(v) for v in scalars["Loss/latent_pose"])
    else:
        assert "Loss/latent_pose" not in scalars


def test_identity_memory_is_the_memoryless_student_on_chunks():
    pytest.importorskip("enc_mem_dec")
    plain, identity = _policy(type="none"), _policy(type="identity")
    assert identity.has_memory
    assert identity.memory.output_dim == identity.action_head[0].in_features
    identity.load_state_dict(plain.state_dict())
    rgb, state = th.rand(4, NUM_ENVS, 9, 16, 16), th.rand(4, NUM_ENVS, 4)
    starts = th.zeros(4, NUM_ENVS, dtype=th.bool)
    with th.no_grad():
        chunk = identity.sequence(rgb, state, starts, burn_in=1)
        steps = th.stack([plain(rgb[t], state[t]) for t in range(1, 4)])
    assert th.allclose(chunk, steps, atol=1e-6)


@pytest.mark.parametrize(
    "memory",
    [{"type": "none"}, {"type": "gru", "hidden_dim": 8}],
    ids=lambda m: m["type"],
)
def test_visual_policy_exposes_the_latent_pose_of_the_decoder(memory):
    if memory["type"] != "none":
        pytest.importorskip("enc_mem_dec")
    env = _CountingEnv()
    images = {"rgb": th.rand(env.num_envs, 9, 16, 16)}
    env.get_modality_observations = lambda modalities: {
        m: images["rgb"][:, 3 * k : 3 * k + 3] for k, m in enumerate(modalities)
    }
    env.manipulator = SimpleNamespace(get_tcp_pose=lambda: th.rand(env.num_envs, 4))

    with_decoder = make_visual_bc_policy(
        env=env, bc_policy=_policy(0.5, **memory), state_obs=["tcp_pose"]
    )
    actions = with_decoder(None)
    assert actions.shape == (env.num_envs, 2)
    assert with_decoder.latent_pose.shape == (env.num_envs, 4)

    without = make_visual_bc_policy(
        env=env, bc_policy=_policy(**memory), state_obs=["tcp_pose"]
    )
    without(None)
    assert without.latent_pose is None
