"""Reproducible random dataset splitting."""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

from torch.utils.data import Dataset, Subset

from .base import BaseSplitter


class RandomSplitter(BaseSplitter):
    def __init__(
        self,
        train_ratio: float = 0.75,
        valid_ratio: float = 0.125,
        test_ratio: float = 0.125,
        seed: Optional[int] = None,
    ) -> None:
        self.train_ratio = float(train_ratio)
        self.valid_ratio = float(valid_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = seed
        self._validate_ratios()

    def _validate_ratios(self) -> None:
        ratios = (self.train_ratio, self.valid_ratio, self.test_ratio)
        if any(ratio < 0 or ratio > 1 for ratio in ratios):
            raise ValueError("split ratios must be between 0 and 1")
        if abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError("train_ratio, valid_ratio, and test_ratio must sum to 1")

    def split_indices(
        self, size: int, *, test_ratio: Optional[float] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        if size < 0:
            raise ValueError("size must be non-negative")
        effective_test = self.test_ratio if test_ratio is None else float(test_ratio)
        if effective_test < 0 or effective_test > 1:
            raise ValueError("test_ratio must be between 0 and 1")
        remaining = 1.0 - effective_test
        if remaining == 0:
            train_ratio = 0.0
        else:
            train_ratio = self.train_ratio / (self.train_ratio + self.valid_ratio)
        valid_ratio = 1.0 - train_ratio

        indices = list(range(size))
        random.Random(self.seed).shuffle(indices)
        test_len = int(size * effective_test)
        valid_len = int((size - test_len) * valid_ratio)
        train_len = size - test_len - valid_len
        train = indices[:train_len]
        valid = indices[train_len : train_len + valid_len]
        test = indices[train_len + valid_len :]
        return train, valid, test

    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        train, valid, test = self.split_indices(len(dataset))
        return Subset(dataset, train), Subset(dataset, valid), Subset(dataset, test)
