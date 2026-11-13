import torch as th

from eval import run_eval

NUM_ENVS = 2


class _Env:
    num_envs = NUM_ENVS

    def __init__(self):
        self.actions: list[th.Tensor] = []

    def step(self, actions: th.Tensor):
        self.actions.append(actions)
        return th.zeros(NUM_ENVS, 3), th.ones(NUM_ENVS), th.zeros(NUM_ENVS), {}


def test_run_eval_steps_the_policy_without_the_autograd_graph():
    env = _Env()
    weight = th.ones(3, 2, requires_grad=True)
    modes = []

    def policy(obs: th.Tensor) -> th.Tensor:
        modes.append(th.is_inference_mode_enabled())
        return obs @ weight

    metrics = run_eval(
        env=env,
        policy=policy,
        max_steps=4,
        obs=th.zeros(NUM_ENVS, 3),
        device=th.device("cpu"),
    )

    assert modes == [True] * 4
    assert not any(a.requires_grad for a in env.actions)
    assert metrics["mean_reward"] == 4.0
    assert metrics["mean_episode_length"] == 4.0
