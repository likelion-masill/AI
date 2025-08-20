# app/logging_config.py
import logging
from logging.config import dictConfig

def setup_logging():
    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "std": {"format": "%(asctime)s %(levelname)s [%(name)s] %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "std",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["console"], "level": "INFO"},
            "uvicorn.error": {"handlers": ["console"], "level": "INFO"},
            "uvicorn.access": {"handlers": ["console"], "level": "INFO"},
            # 너 서비스용 로거
            "app.faiss": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
        },
        "root": {"handlers": ["console"], "level": "INFO"},
    })
