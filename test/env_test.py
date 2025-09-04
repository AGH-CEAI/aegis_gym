import numpy as np
from aegis_gym.envs import AegisReacherEnv

def test_env_AegisReacherEnv():
    kwargs = {"reward_type": "dense", "control_type": "joints"}
    env = AegisReacherEnv(**kwargs)  # noqa: F841


def test_reset_AegisReacherEnv():
    env = AegisReacherEnv()
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
        assert isinstance(obs, np.ndarray), (
            "reset() should return a numpy ndarray as the first value"
        )
        assert isinstance(info, dict), (
            "reset() should return a dict as the second value"
        )
    else:
        assert isinstance(result, np.ndarray), "reset() should return a numpy ndarray"


def test_step_AegisReacherEnv():
    env = AegisReacherEnv()
    env.reset()
    env.step(np.zeros(env.num_actions, dtype=np.float32))


def test_render_AegisReacherEnv():
    env = AegisReacherEnv()
    env.reset()
    env.render()
