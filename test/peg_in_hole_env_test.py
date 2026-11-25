import math
from types import SimpleNamespace

import pytest
import torch as th

from aegis_gym.aux.geom import transform_quat_by_quat
from aegis_gym.aux.pih_assets import PihAssetsCfg
from aegis_gym.aux.pih_assets.assets import CACHE_ENV_VAR
from aegis_gym.config.config_manager import ConfigManager
from aegis_gym.envs import PegInHoleEnv
from aegis_gym.envs.objects import ObjectType
from aegis_gym.envs.peg_in_hole_env import quat_to_yaw, wrap_symmetric, yaw_to_quat
from aegis_gym.envs.scene.base_scene import BaseScene

NUM_ENVS = 4
TABLE_TOP_Z = 0.818 - 0.821


class _Manipulator:
    """Kinematic TCP: the twist moves it exactly, the resets reach the pose exactly."""

    def __init__(self, num_envs: int, policy_dt: float, lin: float, ang: float):
        self.tcp = th.zeros(num_envs, 7)
        self.tcp[:, 3] = 1.0
        self.wrench = th.zeros(num_envs, 6)
        self.policy_dt, self.lin, self.ang = policy_dt, lin, ang
        self.twists = []
        self.reset_open_gripper = []

    def get_tcp_pose(self) -> th.Tensor:
        return self.tcp.clone()

    def get_ft_wrench_compensated(self) -> th.Tensor:
        return self.wrench.clone()

    def ctrl_gripper_close(self, envs_idx) -> None: ...

    def ctrl_go_to_home(self, envs_idx) -> None: ...

    def ctrl_reset_to_pose(self, pose, open_gripper, envs_idx) -> None:
        self.reset_open_gripper.append(open_gripper)
        self.tcp[envs_idx] = pose

    def ctrl_apply_vel_action(self, twist: th.Tensor, open_gripper) -> None:
        self.twists.append(twist.clone())
        self.tcp[:, :3] += twist[:, :3] * self.lin * self.policy_dt
        yaw = twist[:, 5] * self.ang * self.policy_dt
        self.tcp[:, 3:] = transform_quat_by_quat(yaw_to_quat(yaw), self.tcp[:, 3:])


class _Object:
    def __init__(self, properties, scene: "_Scene"):
        self.properties, self.scene = properties, scene
        self.pose = th.tensor(properties.pose).repeat(scene.num_envs, 1)
        self.attached = False

    def get_pose(self) -> th.Tensor:
        if self.attached:  # welded, follows the TCP in the nominal grasp
            return self.scene.env._tcp_to_peg_pose(self.scene.manipulator.tcp)
        return self.pose.clone()

    def set_pose(self, pose: th.Tensor, envs_idx) -> None:
        self.pose[envs_idx] = pose


class _Scene(BaseScene):
    def __init__(self, num_envs: int, policy_dt: float, lin: float, ang: float):
        super().__init__(device=th.device("cpu"))
        self.num_envs = num_envs
        self.manipulator = _Manipulator(num_envs, policy_dt, lin, ang)
        self.objects, self.attached = [], []
        self.peg_pose_at_attach = self.tcp_at_attach = None

    def get_policy_dt(self) -> float:
        return self.manipulator.policy_dt

    def add_entity(self, entity: ObjectType, properties) -> _Object:
        obj = _Object(properties, self)
        self.objects.append(obj)
        return obj

    def add_manipulator(self, cfg) -> None: ...

    def add_preview_camera(self, pos, lookat) -> None: ...

    def attach_object_to_tcp(self, obj: _Object) -> None:
        self.peg_pose_at_attach = obj.get_pose()
        self.tcp_at_attach = self.manipulator.get_tcp_pose()
        obj.attached = True
        self.attached.append(obj)

    def _build(self) -> None: ...

    def _get_manipulator(self) -> _Manipulator:
        return self.manipulator

    def _shutdown(self) -> None: ...

    def _update_state(self) -> None: ...

    def pre_step(self) -> None: ...

    def step(self) -> None: ...

    def get_available_randomizations(self) -> frozenset:
        return frozenset()


def _env(env_dict: dict | None = None, dr: bool = False) -> PegInHoleEnv:
    env_specific = ConfigManager._deep_update(
        base=PegInHoleEnv.get_default_env_specific_dict(), override=env_dict or {}
    )
    cfg_env = SimpleNamespace(
        num_envs=NUM_ENVS,
        num_obs=13,
        num_actions=4,
        visualize_camera=False,
        visualize_cell=True,
        image_resolution=(64, 64),
        cameras_setup="default",
        table_size=[0.55, 0.84, 0.818],
        workbench_size=[0.64, 1.0, 0.821],
        ctrl_dt=0.004,
        policy_dt=0.04,
        episode_length_s=10.0,
        action_max_linear_speed=0.02,
        action_max_angular_speed=0.25,
        env_specific_dict=env_specific,
    )
    cfg = SimpleNamespace(
        env_cfg=cfg_env,
        dr_cfg=SimpleNamespace(enabled=dr),
        robot_cfg=None,
        get_device=lambda: th.device("cpu"),
    )
    scene = _Scene(NUM_ENVS, policy_dt=0.04, lin=0.02, ang=0.25)
    # the peg stub follows the TCP through the env's grasp transform
    original_setup = PegInHoleEnv._setup_scene

    def setup_scene(env: PegInHoleEnv, cfg) -> None:
        scene.env = env
        original_setup(env, cfg)

    PegInHoleEnv._setup_scene = setup_scene
    try:
        return PegInHoleEnv(scene=scene, cfg=cfg)
    finally:
        PegInHoleEnv._setup_scene = original_setup


@pytest.fixture(autouse=True)
def _assets_cache(tmp_path, monkeypatch):
    """The assets generated for the `asset_overrides` go to a temporary cache."""
    monkeypatch.setenv(CACHE_ENV_VAR, str(tmp_path))


# == helpers ==


def test_yaw_helpers():
    yaw = th.tensor([0.3, -2.0, 3.0])
    assert quat_to_yaw(yaw_to_quat(yaw)).tolist() == pytest.approx(yaw.tolist())
    # the square peg is the same every 90 deg
    wrapped = wrap_symmetric(th.tensor([math.radians(80.0), math.radians(-100.0)]), 4)
    assert th.rad2deg(wrapped).tolist() == pytest.approx([-10.0, -10.0], abs=1e-4)


# == env ==


def test_default_env_dict_is_resolved_by_the_config():
    resolved = ConfigManager._resolve_env_specific_dict(
        env_name="peg_in_hole",
        given={"asset_overrides": {"board": {"clearances": [0.003, 0.001]}}},
    )
    assert resolved["asset_overrides"] == {"board": {"clearances": [0.003, 0.001]}}
    assert resolved["active_slots"] == [0]


def test_env_reads_the_nominal_assets():
    env = _env()
    nominal = PihAssetsCfg.from_yaml()

    # board top on the table, slot inlets in a row along its X, centred on `board_xy`
    assert env.board_top_z == pytest.approx(TABLE_TOP_Z + nominal.board.thickness)
    n, pitch = len(nominal.board.clearances), nominal.board.slot_pitch
    expected = th.tensor(
        [[0.47 + (k - (n - 1) / 2) * pitch, 0.0, env.board_top_z] for k in range(n)]
    )
    assert th.allclose(env.slots_pos, expected, atol=1e-6)
    # the assets give the total clearance, the env works per side
    assert env.slots_clearance.tolist() == pytest.approx(
        [c / 2 for c in nominal.board.clearances]
    )
    assert env.clearance.tolist() == pytest.approx(
        [nominal.board.clearances[0] / 2] * NUM_ENVS
    )
    assert env.seated_depth == pytest.approx(nominal.board.depth)
    # the tip is the peg origin, the TCP holds the grasp point above it
    assert env.peg_tips.tolist() == [[0.0, 0.0, 0.0]]
    grasp_z = nominal.peg.length - nominal.peg.grasp_from_top
    assert env.grasp_offset_t.tolist() == pytest.approx([0.0, 0.0, grasp_z])
    assert env.yaw_symmetry == 4
    assert env._asset_paths()[0] == PegInHoleEnv._ASSETS_DIR / "peg.urdf"


def test_asset_overrides_give_the_slots_along_the_board(tmp_path):
    clearances = [0.004, 0.002]
    env = _env(
        env_dict={
            "asset_overrides": {"board": {"clearances": clearances}},
            "board_yaw_deg": 90.0,
            "active_slots": [1],
        }
    )
    pitch = PihAssetsCfg.from_yaml().board.slot_pitch

    # the slots are along the board's X, the world's Y after the 90 deg yaw
    expected = th.tensor([[0.47, -pitch / 2], [0.47, pitch / 2]])
    assert th.allclose(env.slots_pos[:, :2], expected, atol=1e-6)
    assert env.slots_clearance.tolist() == pytest.approx([0.002, 0.001])
    assert (env.slot_idx == 1).all()
    # generated into the cache, not the nominal files
    peg_urdf, board_urdf = env._asset_paths()
    assert peg_urdf.is_relative_to(tmp_path) and board_urdf.exists()


def test_legs_grid_sets_the_tips_and_the_yaw_symmetry():
    env = _env(
        env_dict={
            # wider slot pitch: two legs at the nominal 0.06 leave no wall between slots,
            # which `PihAssetsCfg` rejects
            "asset_overrides": {
                "peg": {"legs": {"grid": [2, 1], "length": 0.03}},
                "board": {"slot_pitch": 0.1},
            }
        }
    )
    pitch = PihAssetsCfg.from_yaml().peg.legs.pitch

    expected = th.tensor([[-pitch / 2, 0.0, 0.0], [pitch / 2, 0.0, 0.0]])
    assert th.allclose(env.peg_tips, expected)
    assert env.yaw_symmetry == 2
    assert env.seated_depth == pytest.approx(min(0.03, env.assets_cfg.board.depth))


def test_reset_doesnt_close_the_fingers_into_the_peg():
    env = _env()

    assert env.manipulator.reset_open_gripper
    assert all(g is None for g in env.manipulator.reset_open_gripper)


def test_peg_is_attached_in_the_grasp_of_the_start_pose():
    env = _env()
    scene = env._scene

    assert scene.attached == [env.peg]
    # the peg was placed into the grasp before the weld was made
    expected = env._tcp_to_peg_pose(scene.tcp_at_attach)
    assert th.allclose(scene.peg_pose_at_attach, expected, atol=1e-6)


def test_tcp_and_peg_poses_round_trip():
    # grasp off the grasp point of assets, to cover `grasp_offset` too
    env = _env(env_dict={"grasp_offset": [0.001, -0.002, 0.01]})
    peg = th.cat([th.tensor([[0.5, 0.1, 0.2]]), yaw_to_quat(th.tensor([0.4]))], dim=-1)
    grasp_z = (
        PihAssetsCfg.from_yaml().peg.length
        - PihAssetsCfg.from_yaml().peg.grasp_from_top
    )
    assert env.grasp_offset_t.tolist() == pytest.approx([0.001, -0.002, grasp_z + 0.01])

    tcp = env._peg_to_tcp_pose(peg)
    back = env._tcp_to_peg_pose(tcp)

    assert th.allclose(back[:, :3], peg[:, :3], atol=1e-6)
    assert quat_to_yaw(back[:, 3:]).item() == pytest.approx(0.4, abs=1e-5)
    # the TCP faces down, it holds an upright peg
    assert env.tilt.max().item() == pytest.approx(0.0, abs=1e-3)


def test_reset_starts_above_the_hole_within_the_ranges():
    env = _env()
    start = env.get_default_env_specific_dict()["start"]

    assert (env.depth <= -start["height_range"][0] + 1e-6).all()
    assert (env.depth >= -start["height_range"][1] - 1e-6).all()
    assert (env.tip_rel[:, :2].abs() <= start["xy_range"] + 1e-6).all()
    assert (env.yaw_err.abs() <= math.radians(start["yaw_range_deg"]) + 1e-5).all()
    assert (env.slot_idx == 0).all()


def test_seeded_reset_is_reproducible_and_without_the_hole_noise():
    env = _env(dr=True)
    seeds = [11, 12, 13, 14]
    idx = th.arange(NUM_ENVS)
    env.reset_seeded(idx, seeds)
    first = env.tip_rel.clone()
    env.reset_seeded(idx, seeds)

    assert th.equal(env.tip_rel, first)
    assert (env.hole_obs_offset == 0).all()


def test_hole_noise_shifts_the_observed_tip_the_other_way():
    env = _env()
    obs = env.get_observations()["policy"]
    env.hole_obs_offset[:, 0] = 0.002

    shifted = env.get_observations()["policy"]

    assert (shifted[:, 0] - obs[:, 0]).tolist() == pytest.approx([-0.002] * NUM_ENVS)


def test_the_actor_sees_no_force_the_critic_does():
    env = _env()
    env.manipulator.wrench[:, 2] = 25.0
    env._obs_cache_clear()
    env._update_insertion_state(env.peg.get_pose())

    obs = env.get_observations()

    # the tip (3), the yaw (2), the TCP velocity (4) and the previous action (4)
    assert obs["policy"].shape == (NUM_ENVS, 13)
    assert not (obs["policy"] == 25.0).any()
    assert (obs[PegInHoleEnv.PRIVILEGED_OBS_GROUP][:, -1] == 25.0).all()


def test_observation_groups():
    env = _env()
    obs = env.get_observations()

    assert obs["policy"].shape == (NUM_ENVS, 13)
    assert obs[PegInHoleEnv.PRIVILEGED_OBS_GROUP].shape == (NUM_ENVS, 12)


def test_seated_peg_is_a_success_with_zero_keypoint_distance():
    env = _env()
    hole = env.hole_pos[0]
    tip = th.tensor([hole[0], hole[1], hole[2] - env.seated_depth])
    peg_pos = tip - env.peg_insertion_point
    peg = th.cat([peg_pos, yaw_to_quat(th.tensor(env.board_yaw))[None][0]])[None]

    env._update_insertion_state(peg.expand(NUM_ENVS, 7).clone())

    assert env.keypoint_dist[0].item() == pytest.approx(0.0, abs=1e-6)
    assert env.depth[0].item() == pytest.approx(env.seated_depth)
    assert env._reward_depth_progress()[0].item() == pytest.approx(1.0)
    assert env._reward_keypoints_fine()[0].item() == pytest.approx(0.5)


def test_depth_progress_is_gated_by_the_yaw():
    env = _env()
    env.depth[:] = env.success_depth
    env.xy_err[:] = 0.0
    env.yaw_err[:] = th.tensor([0.0, math.radians(5.0), math.radians(15.0), -0.5])

    assert env._reward_depth_progress().tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_actions_keep_the_tips_in_the_workspace():
    # the hole frame's X is the world's Y
    env = _env(env_dict={"board_yaw_deg": 90.0})
    # move the tips onto the -X workspace edge of the hole frame
    tcp = env.manipulator.get_tcp_pose()
    tip = env._tips_in_hole_frame(env._tcp_to_peg_pose(tcp)).mean(dim=1)
    shift = th.zeros(NUM_ENVS, 2)
    shift[:, 0] = env.workspace_low[0] - tip[:, 0]
    env.manipulator.tcp[:, :2] += env._rotate_xy(shift, env.hole_yaw)
    tcp = env.manipulator.get_tcp_pose()

    # pushing further out along the hole -X (world -Y) is cut off, inwards is not
    out = th.tensor([[0.0, -1.0, 0.0]]).repeat(NUM_ENVS, 1)
    assert env._limit_action_to_workspace(out, tcp).abs().max().item() < 1e-4
    assert th.allclose(env._limit_action_to_workspace(-out, tcp), -out, atol=1e-5)


def test_oracle_policy_inserts_the_peg():
    env = _env()
    successes = th.zeros(NUM_ENVS, dtype=th.bool)
    for _ in range(env.max_episode_length):
        _, _, dones, extras = env.step(env.scripted_action())
        successes |= extras["success"] & dones
        if successes.all():
            break

    assert successes.all()
    episode = env.extras["episode"]
    assert episode["success_rate"].tolist() == [1.0] * len(episode["success_rate"])


def test_force_limit_ends_the_episode_with_the_penalty():
    env = _env()
    env.manipulator.wrench[1, 2] = env.force_limit + 1.0

    _, reward, dones, extras = env.step(th.zeros(NUM_ENVS, 4))

    assert dones.tolist() == [False, True, False, False]
    assert extras["force_limit"].tolist() == [False, True, False, False]
    assert reward[1] < -40.0
    assert env.extras["episode"]["force_limit_rate"].tolist() == [1.0]


def test_time_out():
    env = _env()
    env.episode_length_buf[2] = env.max_episode_length

    _, _, dones, extras = env.step(th.zeros(NUM_ENVS, 4))

    assert dones.tolist() == [False, False, True, False]
    assert extras["time_outs"].tolist() == [0.0, 0.0, 1.0, 0.0]


def test_contact_penalty_ramps_up():
    env = _env()
    env.force_norm[:] = env.force_contact_threshold + 2.0
    env.global_step = 0
    assert env._reward_contact_penalty().abs().max().item() == 0.0
    env.global_step = env.contact_penalty_ramp_steps // 2
    assert env._reward_contact_penalty()[0].item() == pytest.approx(-1.0)
    env.global_step = 10 * env.contact_penalty_ramp_steps
    assert env._reward_contact_penalty()[0].item() == pytest.approx(-2.0)


def test_invalid_active_slots_are_rejected():
    with pytest.raises(ValueError):
        _env(
            env_dict={"active_slots": [len(PihAssetsCfg.from_yaml().board.clearances)]}
        )
