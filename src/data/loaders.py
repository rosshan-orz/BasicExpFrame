"""DataLoader construction kept independent from Lightning."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.core.seeds import worker_init_fn


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    seed_worker: bool = True,
    pin_memory: Optional[bool] = None,
    **kwargs: Any,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if seed_worker and num_workers else None,
        pin_memory=pin_memory,
        **kwargs,
    )


def build_loaders(
    train: Dataset,
    valid: Dataset,
    test: Dataset,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        build_dataloader(train, shuffle=True, **kwargs),
        build_dataloader(valid, shuffle=False, **kwargs),
        build_dataloader(test, shuffle=False, **kwargs),
    )
