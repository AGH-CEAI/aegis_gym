def test_import_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv  # noqa: F401


def test_env_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

    env = AegisReacherEnv()  # noqa: F841


def test_reset_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

    env = AegisReacherEnv()
    env.reset()


def test_step_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

    env = AegisReacherEnv()
    env.reset()
    env.step(None)


def test_render_AegisReacherEnv():
    from aegis_gym.envs import AegisReacherEnv

    env = AegisReacherEnv()
    env.reset()
    env.render()
