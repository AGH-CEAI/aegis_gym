import logging
import time

from config import setup_logger


def divide(a: float, b: float) -> float:
    print(__name__)
    logger = logging.getLogger(__name__)

    logger.debug("divide(%s, %s)", a, b)

    return a / b


def main() -> None:
    setup_logger("DEBUG")

    logger = logging.getLogger(__name__)

    logger.debug("DEBUG message")
    logger.info("INFO message")
    logger.warning("WARNING message")
    logger.error("ERROR message")
    logger.critical("error")

    logger.info("-" * 60)

    for i in range(3):
        logger.info("Iteration %d", i)
        time.sleep(0.5)

    logger.info("-" * 60)

    try:
        divide(10, 0)
    except ZeroDivisionError:
        logger.exception("Division failed")

    logger.info("Logger test finished")


if __name__ == "__main__":
    main()
