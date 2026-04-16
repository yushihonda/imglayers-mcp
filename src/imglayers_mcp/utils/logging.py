"""Structured logging helpers. Writes to stderr so stdio MCP protocol stays clean."""

from __future__ import annotations

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s :: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        level_name = os.environ.get("IMGLAYERS_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level_name, logging.INFO))
        logger.propagate = False
    return logger
