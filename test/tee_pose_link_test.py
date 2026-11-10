import json
import math
import socket
import time

import pytest
import torch as th

from aegis_gym.envs.objects.base_objects import ObjectProperties
from aegis_gym.envs.objects.real import ros_grpc_objects, tee_pose_link
from aegis_gym.envs.objects.real.ros_grpc_objects import RosGrpcMesh, RosGrpcURDF
from aegis_gym.envs.objects.real.tee_pose_link import TeePoseLink

GYM_PORT = 15601
TRACKER_PORT = 15602


@pytest.fixture
def link(monkeypatch):
    link = TeePoseLink(listen_port=GYM_PORT, tracker_port=TRACKER_PORT)
    monkeypatch.setattr(tee_pose_link, "_LINK", link)
    monkeypatch.setattr(ros_grpc_objects, "get_tee_pose_link", lambda: link)
    yield link
    link.close()


@pytest.fixture
def tracker():
    """The tracker's end: sends poses to the gym, receives the overlay poses."""
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", TRACKER_PORT))
    rx.settimeout(1.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(x, y, yaw):
        packet = {
            "x": x,
            "y": y,
            "yaw": yaw,
            "z": -0.003,
            "latency": 0.05,
            "n_markers": 2,
        }
        tx.sendto(json.dumps(packet).encode(), ("127.0.0.1", GYM_PORT))

    def receive():
        return json.loads(rx.recvfrom(4096)[0])

    yield send, receive
    rx.close()


def _wait_until(predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)


def _properties(pose):
    return ObjectProperties(dims=(0.2, 0.2, 0.04), pose=pose)


def test_tracked_tee_pose(link, tracker):
    send, _ = tracker
    send(0.5, 0.1, 0.3)
    tee = RosGrpcURDF(
        _properties((0.47, 0.0, -0.003, 1.0, 0.0, 0.0, 0.0)), th.device("cpu")
    )
    tee.create()
    send(0.52, -0.05, -2.0)
    _wait_until(lambda: link.latest().x == 0.52)
    pose = tee.get_pose()
    assert pose.shape == (1, 7)
    expected = [0.52, -0.05, -0.003, math.cos(-1.0), 0.0, 0.0, math.sin(-1.0)]
    assert th.allclose(pose[0], th.tensor(expected), atol=1e-6)


def test_tee_spawn_request_reaches_the_tracker(link, tracker, monkeypatch):
    send, receive = tracker
    send(0.5, 0.1, 0.3)
    tee = RosGrpcURDF(
        _properties((0.47, 0.0, -0.003, 1.0, 0.0, 0.0, 0.0)), th.device("cpu")
    )
    tee.create()
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    yaw = 2.5
    tee.set_pose(
        th.tensor([[0.45, 0.2, 0.017, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]])
    )
    msg = receive()
    assert msg["kind"] == "spawn"
    assert msg["x"] == pytest.approx(0.45) and msg["yaw"] == pytest.approx(
        yaw, abs=1e-6
    )


def test_goal_marker_pose_reaches_the_tracker(link, tracker):
    _, receive = tracker
    yaw = -1.2
    goal = RosGrpcMesh(
        _properties(
            (0.47, 0.0, -0.002, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        ),
        th.device("cpu"),
    )
    goal.create()
    msg = receive()
    assert msg["kind"] == "goal"
    assert (msg["x"], msg["y"]) == pytest.approx((0.47, 0.0))
    assert msg["yaw"] == pytest.approx(yaw, abs=1e-6)


def test_tee_requires_the_tracker(link):
    tee = RosGrpcURDF(
        _properties((0.47, 0.0, -0.003, 1.0, 0.0, 0.0, 0.0)), th.device("cpu")
    )
    link.wait_for_pose = lambda timeout_s: None
    with pytest.raises(RuntimeError, match="tee_tracker"):
        tee.create()


def test_untracked_tee_needs_no_tracker(monkeypatch):
    def no_link():
        raise AssertionError("an untracked tee must not open the tracker link")

    monkeypatch.setattr(ros_grpc_objects, "get_tee_pose_link", no_link)
    monkeypatch.setattr(RosGrpcURDF, "tracked", False)
    tee = RosGrpcURDF(_properties([0.5, 0.0, 0.0, 1, 0, 0, 0]), th.device("cpu"))
    tee.create()
    assert tee.get_pose().shape == (1, 7)
    spawn = th.tensor([[0.45, -0.05, 0.0, 1.0, 0.0, 0.0, 0.0]])
    tee.set_pose(spawn)
    assert th.allclose(tee.get_pose(), spawn)
