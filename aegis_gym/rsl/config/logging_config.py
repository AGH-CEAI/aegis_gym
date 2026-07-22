import logging
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[0m",  # Gray
        logging.INFO: "\033[0m",  # Default
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bright red
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logger(level: str = "INFO") -> None:
    log_directory = Path("logs")
    log_directory.mkdir(parents=True, exist_ok=True)

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s|%(filename)s:%(lineno)d | %(message)s"
    )

    date_format = "%H:%M:%S"

    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level.upper())
    console_handler.setFormatter(
        ColoredFormatter(
            fmt=log_format,
            datefmt=date_format,
        )
    )

    file_handler = logging.FileHandler(
        log_directory / "app.log",
        encoding="utf-8",
    )
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(
        logging.Formatter(
            fmt=log_format,
            datefmt=date_format,
        )
    )

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
