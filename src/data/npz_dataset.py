"""Small, conventional NPZ dataset used by examples and smoke tests."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from .base_dataset import BaseDataset
from .registry import DATASET_REGISTRY


@DATASET_REGISTRY.register("NpzDataset")
class NpzDataset(BaseDataset):
    """Read arrays from an NPZ file using configurable input/target keys."""

    def __init__(
        self,
        file_path,
        transform=None,
        input_key: str = "inputs",
        target_key: str = "targets",
    ) -> None:
        super().__init__(file_path=file_path, transform=transform)
        with np.load(self.file_path, allow_pickle=False) as data:
            keys = set(data.files)
            input_key = input_key if input_key in keys else ("x" if "x" in keys else input_key)
            target_key = target_key if target_key in keys else ("y" if "y" in keys else target_key)
            if input_key not in keys or target_key not in keys:
                raise KeyError(
                    f"NPZ must contain input/target arrays; found {sorted(keys)}"
                )
            self.inputs = torch.as_tensor(data[input_key])
            if self.inputs.is_floating_point():
                self.inputs = self.inputs.float()
            self.targets = torch.as_tensor(data[target_key])
        if len(self.inputs) != len(self.targets):
            raise ValueError("input and target arrays must have equal length")

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int):
        sample = {"inputs": self.inputs[index], "targets": self.targets[index]}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
