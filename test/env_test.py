import numpy as np


def test_import_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv  # noqa: F401


def test_env_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

    kwargs = {"reward_type": "dense", "control_type": "joints"}
    env = AegisReacherEnv(**kwargs)  # noqa: F841


def test_reset_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

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
    from aegis_gym.envs import AegisReacherEnv

    env = AegisReacherEnv()
    env.reset()
    env.step(np.zeros(env.num_actions, dtype=np.float32))


def test_render_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

    env = AegisReacherEnv()
    env.reset()
    env.render()
