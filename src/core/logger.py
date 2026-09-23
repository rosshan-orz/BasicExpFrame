"""Lightweight console and file logger used outside Lightning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union


class BaseLogger:
    """A small wrapper around Python's logging module."""

    def __init__(
        self,
        log_dir: Union[str, Path],
        name: Optional[str] = None,
        level: int = logging.INFO,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = f"basic_exp_frame.{self.log_dir.name or 'experiment'}"

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)

        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        self.logger.addHandler(console)

        file_path = self.log_dir / "train.log"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)

        self.info(f"Logging initialized. Log directory: {self.log_dir}")

    def debug(self, msg: str, *args, **kwargs) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        self.logger.error(msg, *args, **kwargs)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
