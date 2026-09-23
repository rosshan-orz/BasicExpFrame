"""Deterministic sequential dataset splitting."""

from __future__ import annotations

from typing import List, Optional, Tuple

from torch.utils.data import Dataset, Subset

from .base import BaseSplitter


class SequentialSplitter(BaseSplitter):
    def __init__(
        self,
        train_ratio: float = 0.75,
        valid_ratio: float = 0.125,
        test_ratio: float = 0.125,
        seed: Optional[int] = None,
        ratio: Optional[float] = None,
    ) -> None:
        # ``ratio`` is retained as a compatibility alias for older configs.
        if ratio is not None:
            train_ratio = ratio
            remaining = 1.0 - float(ratio)
            valid_ratio = remaining / 2
            test_ratio = remaining / 2
        self.train_ratio = float(train_ratio)
        self.valid_ratio = float(valid_ratio)
        self.test_ratio = float(test_ratio)
        self.seed = seed
        ratios = (self.train_ratio, self.valid_ratio, self.test_ratio)
        if any(value < 0 or value > 1 for value in ratios) or abs(sum(ratios) - 1) > 1e-8:
            raise ValueError("split ratios must be between 0 and 1 and sum to 1")

    def split_indices(
        self, size: int, *, test_ratio: Optional[float] = None
    ) -> Tuple[List[int], List[int], List[int]]:
        effective_test = self.test_ratio if test_ratio is None else float(test_ratio)
        if effective_test < 0 or effective_test > 1:
            raise ValueError("test_ratio must be between 0 and 1")
        test_len = int(size * effective_test)
        remaining = size - test_len
        denominator = self.train_ratio + self.valid_ratio
        valid_fraction = self.valid_ratio / denominator if denominator else 0.0
        valid_len = int(remaining * valid_fraction)
        train_len = size - valid_len - test_len
        return (
            list(range(train_len)),
            list(range(train_len, train_len + valid_len)),
            list(range(train_len + valid_len, size)),
        )

    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        train, valid, test = self.split_indices(len(dataset))
        return Subset(dataset, train), Subset(dataset, valid), Subset(dataset, test)
