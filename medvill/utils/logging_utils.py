"""Structured logging helpers."""
from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = "") -> None:
    parts = [f"{prefix}{k}={v:.4f}" if isinstance(v, float) else f"{prefix}{k}={v}"
             for k, v in metrics.items()]
    logger.info("  ".join(parts))
