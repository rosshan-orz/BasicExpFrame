"""Configuration loading helpers based on YAML and python-box."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from box import Box


class ConfigParser:
    """Parser for YAML-based experiment configuration."""

    @staticmethod
    def load(path: Union[str, Path]) -> Box:
        """Load a YAML file into a frozen Box object."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"Config file is empty: {config_path}")

        if not isinstance(raw, dict):
            raise ValueError(
                f"Config root must be a mapping, got {type(raw).__name__}: {config_path}"
            )

        return Box(raw, frozen_box=True, default_box=False)


def load_config(path: Union[str, Path]) -> Box:
    """Convenience alias for :meth:`ConfigParser.load`."""
    return ConfigParser.load(path)
