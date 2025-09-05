import logging
import os
from logging.handlers import RotatingFileHandler


def get_logger(
    name: str = "app",
    file_level: int = logging.WARNING,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    log_file: str = "logs/app.log",
) -> logging.Logger:
    """
    Logger for non-web modules (core, utils, etc.)

    ENV behavior:
    - ENV=dev → logs to stdout, INFO level
    - ENV=prod → logs WARNING+ to rotating file
    """
    env = os.getenv("ENV", "dev").lower()
    console_level = logging.INFO if env == "dev" else file_level

    logger = logging.getLogger(name)
    logger.setLevel(console_level)

    if not logger.hasHandlers():
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s: %(lineno)d - %(levelname)s - %(message)s"
        )

        # Dev: stdout
        if env == "dev":
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(console_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
        else:
            # Prod: rotating file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
