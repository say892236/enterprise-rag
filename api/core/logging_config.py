# api/core/logging_config.py

"""
日志配置模块。

作用：
1. 统一配置项目日志格式。
2. 同时输出到控制台和 logs/app.log。
3. 对第三方库日志降噪。
"""

import logging
import logging.config
from pathlib import Path

from config.settings import (
    LOG_LEVEL,
    LOG_DIR,
    LOG_FILE,
)


# 这些第三方库 INFO 日志太多，统一降到 WARNING
NOISY_LOGGERS = [
    "httpx",
    "httpcore",
    "huggingface_hub",
    "transformers",
    "sentence_transformers",
    "urllib3",
    "filelock",
]


def setup_logging() -> None:
    """
    初始化项目日志配置。
    """

    Path(LOG_DIR).mkdir(
        parents=True,
        exist_ok=True,
    )

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "standard": {
                "format": (
                    "%(asctime)s | "
                    "%(levelname)s | "
                    "%(name)s | "
                    "%(message)s"
                ),
                "datefmt": "%Y-%m-%d %H:%M:%S",
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": LOG_LEVEL,
                "formatter": "standard",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": LOG_LEVEL,
                "formatter": "standard",
                "filename": LOG_FILE,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },

        "root": {
            "level": LOG_LEVEL,
            "handlers": ["console", "file"],
        },

        "loggers": {
            "uvicorn": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logging_config)

    # 关键：dictConfig 后再强制设置第三方库日志级别
    # 这样即使第三方库已经提前创建了 logger，也能压住它们的 INFO 日志。
    for logger_name in NOISY_LOGGERS:
        noisy_logger = logging.getLogger(logger_name)
        noisy_logger.setLevel(logging.WARNING)