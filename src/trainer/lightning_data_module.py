"""Lightning DataModule that only forwards prepared datasets."""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset

from src.data.experiment import Experiment
from src.data.loaders import build_dataloader


class ExperimentDataModule(pl.LightningDataModule):
    def __init__(
        self,
        experiment: Optional[Experiment] = None,
        *,
        train: Optional[Dataset] = None,
        valid: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        **loader_kwargs: Any,
    ) -> None:
        super().__init__()
        if experiment is not None:
            train, valid, test = experiment.train, experiment.valid, experiment.test
        if train is None or valid is None or test is None:
            raise ValueError("experiment or train/valid/test datasets are required")
        self.train_dataset = train
        self.valid_dataset = valid
        self.test_dataset = test
        self.loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": num_workers,
            **loader_kwargs,
        }

    def train_dataloader(self) -> DataLoader:
        return build_dataloader(self.train_dataset, shuffle=True, **self.loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return build_dataloader(self.valid_dataset, shuffle=False, **self.loader_kwargs)

    def test_dataloader(self) -> DataLoader:
        return build_dataloader(self.test_dataset, shuffle=False, **self.loader_kwargs)
