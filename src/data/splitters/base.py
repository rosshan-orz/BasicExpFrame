"""Interfaces shared by dataset splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

from torch.utils.data import Dataset


class BaseSplitter(ABC):
    """Split a dataset into train, validation, and test subsets."""

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        raise NotImplementedError

    @abstractmethod
    def split_indices(
        self, size: int, *, test_ratio: Optional[float] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        """Return disjoint train/valid/test indices for ``size`` items."""
        raise NotImplementedError

    def split_sequence(
        self, values: Sequence[str], *, test_ratio: Optional[float] = None
    ) -> Tuple[List[str], List[str], List[str]]:
        indices = self.split_indices(len(values), test_ratio=test_ratio)
        return (
            [values[i] for i in indices[0]],
            [values[i] for i in indices[1]],
            [values[i] for i in indices[2]],
        )
