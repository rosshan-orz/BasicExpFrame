"""Base dataset abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional, Union

from torch.utils.data import Dataset

from .sample_type import SampleDict


PathLike = Union[str, Path]


class BaseDataset(Dataset, ABC):
    """Base class for file-backed datasets.

    Subclasses only need to implement ``__len__`` and ``__getitem__``.  The
    constructor validates a supplied file path and stores an optional sample
    transform.  In-memory datasets may pass ``file_path=None``.
    """

    def __init__(
        self,
        file_path: Optional[PathLike] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.file_path = Path(file_path) if file_path is not None else None
        if self.file_path is not None:
            self.validate_file_path(self.file_path)
        self.transform = transform

    @staticmethod
    def validate_file_path(file_path: PathLike) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file or directory not found: {path}")
        return path

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> SampleDict:
        raise NotImplementedError
