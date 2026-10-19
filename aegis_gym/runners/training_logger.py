import statistics
from collections.abc import Callable
from pathlib import Path

from rsl_rl.utils.logger import Logger

from aegis_gym.aux.logging import get_logger

logger = get_logger(__name__)


class TrainingLogger(Logger):
    def __init__(
        self,
        *args,
        save_best_model: Callable[[Path], None] | None = None,
        best_model_skip_iters: int = 0,
        reset_last_layer: Callable[[], None] | None = None,
        reset_last_layer_interval: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._save_best_model = save_best_model
        self._best_model_skip_iters = best_model_skip_iters
        self._reset_last_layer = reset_last_layer
        self._reset_last_layer_interval = reset_last_layer_interval

        self._best_model_reward = float("-inf")
        self._best_model_iter = -1
        self._reset_pending = False

    def log(self, it: int, *args, **kwargs) -> None:
        super().log(it, *args, **kwargs)

        if self.writer is None or self.disable_logs:
            return

        if self._save_best_model is not None and self.rewbuffer:
            self._update_best_model(it, statistics.mean(self.rewbuffer))

        if self._is_reset_due(it):
            if it % self.cfg["save_interval"] == 0:
                self._reset_pending = True
            else:
                self._run_reset_last_layer()

    def save_model(self, path: str, it: int) -> None:
        super().save_model(path, it)
        if self._reset_pending:
            self._reset_pending = False
            self._run_reset_last_layer()

    def _is_reset_due(self, it: int) -> bool:
        interval = self._reset_last_layer_interval
        if self._reset_last_layer is None or interval <= 0:
            return False
        return (it + 1) % interval == 0

    def _run_reset_last_layer(self) -> None:
        assert self._reset_last_layer is not None
        self._reset_last_layer()
        logger.info("Reset the last layer weights")

    def _update_best_model(self, it: int, mean_reward: float) -> None:
        if it < self._best_model_skip_iters or mean_reward <= self._best_model_reward:
            return

        assert self._save_best_model is not None
        assert self.log_dir is not None
        self._save_best_model(Path(self.log_dir) / "model_best.pt")
        self._best_model_reward = mean_reward
        self._best_model_iter = it

        logger.info(
            f"New best model! Iteration = {it}, mean reward = {mean_reward:.2f}"
        )

    @property
    def best_model(self) -> tuple[int, float]:
        return self._best_model_iter, self._best_model_reward
