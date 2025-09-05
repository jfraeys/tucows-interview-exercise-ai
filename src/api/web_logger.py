import logging
import os
from logging.handlers import RotatingFileHandler


def get_web_logger(
    log_file: str = "logs/web.log",
    file_level: int = logging.WARNING,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
) -> logging.Logger:
    """
    Logger for web modules (FastAPI + Uvicorn)

    ENV behavior:
    - ENV=dev → logs to stdout
    - ENV=prod → logs WARNING+ to rotating file, overrides Uvicorn logs
    """
    env = os.getenv("ENV", "dev").lower()
    logger_level = logging.INFO if env == "dev" else file_level

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Web logger
    web_logger = logging.getLogger("web")
    web_logger.setLevel(logger_level)

    if not web_logger.hasHandlers():
        if env == "dev":
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logger_level)
            stream_handler.setFormatter(formatter)
            web_logger.addHandler(stream_handler)
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
            file_handler.setLevel(file_level)
            file_handler.setFormatter(formatter)
            web_logger.addHandler(file_handler)

            # Override Uvicorn loggers
            for uv_logger_name in ["uvicorn.access", "uvicorn.error"]:
                uv_logger = logging.getLogger(uv_logger_name)
                uv_logger.handlers.clear()
                uv_logger.addHandler(file_handler)
                uv_logger.setLevel(file_level)

    return web_logger


def get_uvicorn_logger() -> logging.Logger:
    """
    Returns the Uvicorn logger so messages appear in Uvicorn stdout.

    Use this for console-level logs that you want to see in Uvicorn output.
    """
    return logging.getLogger("uvicorn")
