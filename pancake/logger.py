"""
Simple pancake logger.
----
Usage:
> from pancake.logger import setup_logger
> logger = setup_logger(__name__)  # You can use any name
> logger.debug('Happy logging!')

If you want to adjust the level, use
> logger.setLevel(logging.DEBUG)
where level is one of DEBUG, INFO, WARNING, ERROR or CRITICAL.

----
2021 mauricesvp
"""
import logging


def setup_logger(name: str) -> logging.Logger:
    """Logger setup."""
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(filename="pancake.log")
    stderr_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stderr_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(stderr_handler)
    return logger
