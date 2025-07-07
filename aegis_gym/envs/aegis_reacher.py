from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

Observation = dict[str, np.ndarray]
Info = dict[str, float]
Reward = float
Terminated = bool
Truncated = bool


# TODO: Implement the actual environment logic
class AegisReacherEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, device: str = "cpu", render_mode: str = None):
        super().__init__()
        self.device = device

        self.num_obs = 18
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_obs,), dtype=np.float32
        )
        self.obs_scales = {"dof_pos": 1.0, "dof_vel": 0.1}

        self.num_actions = 6
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.action_scale = 0.5

        self.episode_step = 0.0
        self.episode_start_time = 0.0

        self.actions = np.zeros(self.num_actions)
        self.target_pos = np.zeros(3)

        self.reward_functions = {
            "dist": lambda: 1.0,
            "control": lambda: 1.0,
        }

        self.episode_sums = {key: 0.0 for key in self.reward_functions}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def step(
        self, action: np.ndarray
    ) -> tuple[Observation, Reward, Terminated, Truncated, Info]:
        return self._get_obs(), 0.0, False, False, self._get_info()

    def reset(self, seed=None, options=None) -> tuple[Observation, Info]:
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> Observation:
        return {"obs": np.random.rand(self.num_obs).astype(np.float32)}

    def _get_info(self) -> Info:
        return {"distance": 0.0}

    def render(self) -> Optional[np.ndarray]:
        return None
