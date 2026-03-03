"""
utils.py — Shared utilities: config loading, logging, directory setup
"""

import os
import logging
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML config file and return as dict."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "churn", level: int = logging.INFO) -> logging.Logger:
    """Set up a clean logger with timestamp."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_dirs(config: dict) -> None:
    """Create output directories if they don't exist."""
    dirs = [
        "data/raw",
        "data/processed",
        config["output"]["models_dir"],
        config["output"]["figures_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent
