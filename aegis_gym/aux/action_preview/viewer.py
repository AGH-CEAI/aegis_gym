"""
The action preview window, run in its own process (`ActionPreviewClient`), so drawing it never
delays the control loop: `python -m aegis_gym.aux.action_preview.viewer --fd <socket> ...`.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from aegis_gym.aux.logging import get_logger

from .renderer import ActionPreviewRenderer
from .spec import ActionSpec

_MAX_FPS = 30
_MAX_MESSAGE_SIZE = 65536


class ActionPreviewClient:
    """Starts the action preview window and sends it the actions, without ever blocking."""

    def __init__(
        self,
        spec: ActionSpec,
        max_steps: int,
        policy_dt: float,
        title: str,
        save_dir: Path,
    ) -> None:
        self._logger = get_logger("ActionPreview")
        self._dropped = 0
        parent_sock, child_sock = socket.socketpair(
            socket.AF_UNIX, socket.SOCK_SEQPACKET
        )
        parent_sock.setblocking(False)
        # the package root, the viewer may run from another working directory
        package_root = str(Path(__file__).resolve().parents[3])
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (package_root, env.get("PYTHONPATH")) if p
        )
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aegis_gym.aux.action_preview.viewer",
                f"--fd={child_sock.fileno()}",
                f"--spec={json.dumps(spec.as_dict())}",
                f"--max-steps={max_steps}",
                f"--policy-dt={policy_dt}",
                f"--title={title}",
                f"--save-dir={save_dir}",
            ],
            pass_fds=(child_sock.fileno(),),
            env=env,
        )
        child_sock.close()
        self._sock: socket.socket | None = parent_sock

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def send(
        self,
        step: int,
        raw: list[float],
        applied: list[float] | None,
        done: bool,
        reason: str | None,
    ) -> None:
        if self._sock is None:
            return
        message = {
            "step": step,
            "raw": raw,
            "applied": applied,
            "done": done,
            "reason": reason,
        }
        try:
            self._sock.send(json.dumps(message).encode())
        except BlockingIOError:
            # the viewer is behind, its window shows a gap there
            self._dropped += 1
        except OSError:
            self._logger.info("The action preview window was closed.")
            self.close()

    def close(self) -> None:
        if getattr(self, "_sock", None) is not None:
            self._sock.close()
            self._sock = None
        if getattr(self, "_process", None) is None:
            return
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
        if self._dropped:
            self._logger.warning(
                f"{self._dropped} steps were dropped by the busy action preview window."
            )

    def __del__(self) -> None:
        self.close()


def run_viewer(
    sock: socket.socket,
    renderer: ActionPreviewRenderer,
    save_dir: Path,
) -> None:
    """Shows the actions received on the `sock` until it's closed or `q` is pressed."""
    logger = get_logger("ActionPreview")
    parent_pid = os.getppid()
    sock.setblocking(False)
    window = renderer.title
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    paused = False
    frame = renderer.render()
    while True:
        frame_start = time.monotonic()
        while True:
            try:
                data = sock.recv(_MAX_MESSAGE_SIZE)
            except BlockingIOError:
                break
            if not data:
                return
            message = json.loads(data)
            renderer.add(
                step=message["step"],
                raw=message["raw"],
                applied=message["applied"],
                done=message["done"],
                reason=message["reason"],
            )
        if os.getppid() != parent_pid:
            return
        if not paused:
            frame = renderer.render()
        cv2.imshow(window, frame)

        wait_ms = max(
            1, round(1000 / _MAX_FPS - 1000 * (time.monotonic() - frame_start))
        )
        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            return
        if key == ord(" "):
            paused = not paused
            frame = renderer.render(paused=paused)
        elif key == ord("s"):
            save_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            path = save_dir / f"action_preview_{timestamp}.png"
            cv2.imwrite(str(path), frame)
            logger.info(f"Saved the action preview to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fd", type=int, required=True)
    p.add_argument("--spec", type=str, required=True)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--policy-dt", type=float, required=True)
    p.add_argument("--title", type=str, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    args = p.parse_args()

    renderer = ActionPreviewRenderer(
        spec=ActionSpec.from_dict(json.loads(args.spec)),
        max_steps=args.max_steps,
        policy_dt=args.policy_dt,
        title=args.title,
    )
    sock = socket.socket(fileno=args.fd)
    try:
        run_viewer(sock=sock, renderer=renderer, save_dir=args.save_dir)
    finally:
        sock.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
