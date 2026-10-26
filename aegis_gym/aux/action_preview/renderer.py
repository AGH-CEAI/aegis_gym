from collections.abc import Callable, Sequence

import cv2
import numpy as np

from .spec import ActionSpec

Color = tuple[int, int, int]  # BGR

_BACKGROUND: Color = (255, 255, 255)
_PLOT_BACKGROUND: Color = (250, 250, 250)
_GRID: Color = (215, 215, 215)
_LIMIT: Color = (150, 150, 150)
_TEXT: Color = (40, 40, 40)
_PREVIOUS: Color = (195, 195, 195)
_RAW: Color = (220, 150, 50)
_APPLIED: Color = (20, 110, 235)
_LIMITED: Color = (225, 225, 255)
_LIMITED_LINK: Color = (110, 110, 230)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class ActionPreviewRenderer:
    """
    Draws the actions of the current episode of an environment, step by step: a strip per action
    with the raw (policy) and the applied (after the env's clamping and limits) actions over the
    whole episode, the previous episode in the background and the limited steps highlighted, and
    the joystick panel of the `ActionSpec.joystick` action pair.
    """

    HEADER_HEIGHT = 50
    STRIP_HEIGHT = 84
    FOOTER_HEIGHT = 24
    STRIPS_WIDTH = 780
    LABEL_WIDTH = 92
    VALUE_WIDTH = 140
    JOYSTICK_WIDTH = 300
    JOYSTICK_SIDE = 260
    Y_LIMIT = 1.5

    def __init__(
        self, spec: ActionSpec, max_steps: int, policy_dt: float, title: str
    ) -> None:
        self.spec = spec
        self.max_steps = max(int(max_steps), 1)
        self.policy_dt = policy_dt
        self.title = title
        self._raw = self._empty(self.max_steps + 1)
        self._applied = self._empty(self.max_steps + 1)
        self._has_applied = False
        self._previous: np.ndarray | None = None
        self._episode = 0
        self._last_step = -1
        self._ended = False
        self._last_end_reason: str | None = None

    def add(
        self,
        step: int,
        raw: Sequence[float],
        applied: Sequence[float] | None = None,
        done: bool = False,
        reason: str | None = None,
    ) -> None:
        """
        Adds the `raw` and `applied` actions of the `step` of the episode (0 for its first one).
        A step not following the last one, or one after the `done`, starts the next episode.
        """
        if self._last_step < 0:
            self._episode = 1
        elif self._ended or step <= self._last_step:
            self._start_episode()
        if step >= len(self._raw):
            self._grow(step + 1)
        self._raw[step] = raw
        if applied is not None:
            self._applied[step] = applied
            self._has_applied = True
        self._last_step = step
        self._ended = done
        if done:
            self._last_end_reason = reason or "done"

    def render(self, paused: bool = False) -> np.ndarray:
        """The [H, W, 3] BGR image of the actions."""
        num_actions = len(self.spec.names)
        has_joystick = self.spec.joystick is not None
        width = self.STRIPS_WIDTH + (self.JOYSTICK_WIDTH if has_joystick else 0)
        height = (
            self.HEADER_HEIGHT + num_actions * self.STRIP_HEIGHT + self.FOOTER_HEIGHT
        )
        if has_joystick:
            height = max(
                height, self.HEADER_HEIGHT + self.JOYSTICK_SIDE + 2 * self.FOOTER_HEIGHT
            )
        img = np.full((height, width, 3), _BACKGROUND, dtype=np.uint8)

        self._draw_header(img, paused=paused)
        for i in range(num_actions):
            self._draw_strip(img, i, top=self.HEADER_HEIGHT + i * self.STRIP_HEIGHT)
        self._draw_time_axis(
            img, top=self.HEADER_HEIGHT + num_actions * self.STRIP_HEIGHT
        )
        if has_joystick:
            self._draw_joystick(img, left=self.STRIPS_WIDTH, top=self.HEADER_HEIGHT)
        return img

    def _empty(self, length: int) -> np.ndarray:
        return np.full((length, len(self.spec.names)), np.nan)

    def _grow(self, length: int) -> None:
        length = max(length, 2 * len(self._raw))
        for name in ("_raw", "_applied"):
            values = getattr(self, name)
            grown = self._empty(length)
            grown[: len(values)] = values
            setattr(self, name, grown)

    def _start_episode(self) -> None:
        self._previous = self._shown()[: self._last_step + 1].copy()
        self._raw = self._empty(self.max_steps + 1)
        self._applied = self._empty(self.max_steps + 1)
        self._episode += 1
        self._ended = False

    def _shown(self) -> np.ndarray:
        """The actions of the joystick, the current values and the previous episode."""
        return self._applied if self._has_applied else self._raw

    def _span(self) -> int:
        """The steps of the time axis, the whole episode."""
        previous = len(self._previous) - 1 if self._previous is not None else 0
        return max(self.max_steps, self._last_step, previous)

    def _x(self, steps: np.ndarray | float) -> np.ndarray | float:
        x0, x1 = self.LABEL_WIDTH, self.STRIPS_WIDTH - self.VALUE_WIDTH
        return x0 + np.asarray(steps) * (x1 - x0) / self._span()

    def _draw_header(self, img: np.ndarray, paused: bool) -> None:
        steps = self._last_step + 1
        status = (
            f"{self.title} | episode {self._episode} | step {steps}/{self.max_steps} "
            f"({steps * self.policy_dt:.1f}/{self.max_steps * self.policy_dt:.1f} s)"
        )
        if self._last_end_reason is not None:
            status += f" | last end: {self._last_end_reason}"
        if paused:
            status += " | PAUSED"
        _text(img, status, (8, 18), scale=0.5)

        x, y = 8, 40
        for label, color, thickness in (
            ("raw", _RAW, 1),
            ("applied", _APPLIED, 2),
            ("previous episode", _PREVIOUS, 1),
        ):
            if label == "applied" and not self._has_applied:
                continue
            cv2.line(img, (x, y - 4), (x + 22, y - 4), color, thickness, cv2.LINE_AA)
            x = _text(img, label, (x + 28, y), scale=0.42) + 16
        cv2.rectangle(img, (x, y - 10), (x + 22, y + 2), _LIMITED, -1)
        _text(img, "clamped / limited", (x + 28, y), scale=0.42)
        _text(img, "keys: space pause, s save, q close", (img.shape[1] - 270, y), 0.42)

    def _draw_strip(self, img: np.ndarray, i: int, top: int) -> None:
        x0, x1 = self.LABEL_WIDTH, self.STRIPS_WIDTH - self.VALUE_WIDTH
        y_top, y_bottom = top + 6, top + self.STRIP_HEIGHT - 6

        def y(values: np.ndarray | float) -> np.ndarray | float:
            clipped = np.clip(values, -self.Y_LIMIT, self.Y_LIMIT)
            return y_top + (self.Y_LIMIT - clipped) / (2 * self.Y_LIMIT) * (
                y_bottom - y_top
            )

        cv2.rectangle(img, (x0, y_top), (x1, y_bottom), _PLOT_BACKGROUND, -1)
        raw, applied = self._raw[:, i], self._applied[:, i]
        with np.errstate(invalid="ignore"):
            limited = np.abs(raw) > 1.0
            if self._has_applied:
                limited |= np.abs(raw - applied) > 1e-3
        half_step = 0.5 * (x1 - x0) / self._span()
        for k in np.flatnonzero(limited):
            xk = self._x(k)
            cv2.rectangle(
                img,
                (round(xk - half_step), y_top),
                (round(xk + half_step), y_bottom),
                _LIMITED,
                -1,
            )

        cv2.line(img, (x0, round(y(0.0))), (x1, round(y(0.0))), _GRID, 1)
        for limit in (-1.0, 1.0):
            _dashed_hline(img, x0, x1, round(y(limit)), _LIMIT)
        if self._previous is not None:
            _polyline(img, self._previous[:, i], self._x, y, _PREVIOUS, 1)
        _polyline(img, raw, self._x, y, _RAW, 1)
        if self._has_applied:
            _polyline(img, applied, self._x, y, _APPLIED, 2)
        cv2.rectangle(img, (x0, y_top), (x1, y_bottom), _GRID, 1)

        middle = top + self.STRIP_HEIGHT // 2
        _text(img, self.spec.names[i], (8, middle - 2), scale=0.55)
        if self.spec.units[i]:
            _text(img, f"[{self.spec.units[i]}]", (8, middle + 18), scale=0.42)
        if self._last_step < 0:
            return
        value = self._shown()[self._last_step, i]
        if np.isnan(value):
            return
        command = f"{value * self.spec.scales[i]:+.3f} {self.spec.units[i]}".rstrip()
        _text(img, command, (x1 + 10, middle - 2), scale=0.5)
        _text(img, f"({value:+.2f})", (x1 + 10, middle + 18), scale=0.42)

    def _draw_time_axis(self, img: np.ndarray, top: int) -> None:
        span = self._span()
        for fraction in np.linspace(0.0, 1.0, 5):
            step = fraction * span
            x = round(self._x(step))
            cv2.line(img, (x, top - 6), (x, top - 2), _LIMIT, 1)
            label = f"{step * self.policy_dt:.1f} s"
            _text(img, label, (x - 4 * len(label), top + 14), scale=0.4)

    def _draw_joystick(self, img: np.ndarray, left: int, top: int) -> None:
        i, j = self.spec.joystick
        side = self.JOYSTICK_SIDE
        x0, y0 = left + (self.JOYSTICK_WIDTH - side) // 2, top + 6
        cx, cy = x0 + side / 2, y0 + side / 2

        def point(a: float, b: float) -> tuple[int, int]:
            a, b = np.clip((a, b), -self.Y_LIMIT, self.Y_LIMIT)
            half = side / 2 / self.Y_LIMIT
            return round(cx + a * half), round(cy - b * half)

        cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), _PLOT_BACKGROUND, -1)
        cv2.line(img, (x0, round(cy)), (x0 + side, round(cy)), _GRID, 1)
        cv2.line(img, (round(cx), y0), (round(cx), y0 + side), _GRID, 1)
        cv2.rectangle(img, point(-1.0, 1.0), point(1.0, -1.0), _LIMIT, 1)
        cv2.rectangle(img, (x0, y0), (x0 + side, y0 + side), _GRID, 1)

        if self._previous is not None:
            trail = [
                point(a, b)
                for a, b in self._previous[:, [i, j]]
                if not (np.isnan(a) or np.isnan(b))
            ]
            if len(trail) > 1:
                cv2.polylines(
                    img, [np.array(trail, np.int32)], False, _PREVIOUS, 1, cv2.LINE_AA
                )

        shown, span = self._shown(), self._span()
        for k in range(1, self._last_step + 1):
            a0, b0 = shown[k - 1, [i, j]]
            a1, b1 = shown[k, [i, j]]
            if np.isnan((a0, b0, a1, b1)).any():
                continue
            progress = k / span
            color = (round(255 * (1 - progress)), 0, round(255 * progress))
            cv2.line(img, point(a0, b0), point(a1, b1), color, 2, cv2.LINE_AA)

        if self._last_step >= 0:
            raw = self._raw[self._last_step, [i, j]]
            if not np.isnan(raw).any():
                cv2.circle(img, point(*raw), 6, _RAW, 2, cv2.LINE_AA)
            applied = self._applied[self._last_step, [i, j]]
            if self._has_applied and not np.isnan(applied).any():
                if not np.isnan(raw).any():
                    cv2.line(
                        img, point(*raw), point(*applied), _LIMITED_LINK, 1, cv2.LINE_AA
                    )
                cv2.circle(img, point(*applied), 5, _APPLIED, -1, cv2.LINE_AA)

        names = self.spec.names
        _text(img, f"{names[i]} (right) / {names[j]} (up)", (x0, y0 + side + 18), 0.45)
        _text(img, "trail: start (blue) to end (red)", (x0, y0 + side + 36), 0.4)


def _text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float = 0.45,
    color: Color = _TEXT,
) -> int:
    """Puts the `text` at `org`, returns the x coordinate of its end."""
    cv2.putText(img, text, org, _FONT, scale, color, 1, cv2.LINE_AA)
    (text_width, _), _ = cv2.getTextSize(text, _FONT, scale, 1)
    return org[0] + text_width


def _dashed_hline(
    img: np.ndarray, x0: int, x1: int, y: int, color: Color, dash: int = 4
) -> None:
    for x in range(x0, x1, 2 * dash):
        cv2.line(img, (x, y), (min(x + dash, x1), y), color, 1)


def _polyline(
    img: np.ndarray,
    values: np.ndarray,
    x: Callable[[np.ndarray], np.ndarray],
    y: Callable[[np.ndarray], np.ndarray],
    color: Color,
    thickness: int,
) -> None:
    """Draws the `values` by their step, with gaps at the missing (NaN) ones."""
    steps = np.flatnonzero(~np.isnan(values))
    if steps.size == 0:
        return
    for segment in np.split(steps, np.flatnonzero(np.diff(steps) != 1) + 1):
        points = np.stack([x(segment), y(values[segment])], axis=-1)
        points = np.round(points).astype(np.int32)
        if len(points) == 1:
            cv2.circle(img, tuple(points[0]), thickness, color, -1)
        else:
            cv2.polylines(img, [points], False, color, thickness, cv2.LINE_AA)
