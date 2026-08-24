from __future__ import annotations

import logging
from typing import ClassVar

_INITIALIZED = False

DEFAULT_CONSOLE_LEVEL = logging.INFO
APP_LOGGER_NAME = "aegis_gym"


class ColoredFormatter(logging.Formatter):
    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[90m",
        logging.INFO: "\033[0m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }

    RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        copied_record = logging.makeLogRecord(record.__dict__.copy())

        prefix = f"{APP_LOGGER_NAME}."

        if copied_record.name.startswith(prefix):
            copied_record.short_name = copied_record.name[len(prefix) :]
        else:
            copied_record.short_name = copied_record.name

        color = self.COLORS.get(copied_record.levelno, self.RESET)
        message = super().format(copied_record)

        return f"{color}{message}{self.RESET}"


def lean_format() -> ColoredFormatter:
    fmt = "[%(short_name)s] [%(asctime)s] [%(levelname)s] %(message)s"

    return ColoredFormatter(
        fmt=fmt,
        datefmt="%H:%M:%S",
    )


def terminal_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(lean_format())
    return handler


def setup_logger(
    console_level: int = DEFAULT_CONSOLE_LEVEL,
    force: bool = False,
) -> None:
    global _INITIALIZED

    if _INITIALIZED and not force:
        return

    app_logger = logging.getLogger(APP_LOGGER_NAME)
    app_logger.setLevel(console_level)

    app_logger.propagate = False

    for handler in app_logger.handlers[:]:
        app_logger.removeHandler(handler)
        handler.close()

    app_logger.addHandler(terminal_handler(console_level))

    _INITIALIZED = True


def get_logger(name: str | None = None) -> logging.Logger:

    if name is None or name == APP_LOGGER_NAME or name.startswith(f"{APP_LOGGER_NAME}."):
        logger_name = name
    else:
        logger_name = f"{APP_LOGGER_NAME}.{name}"

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.NOTSET)
    logger.propagate = True

    return logger
