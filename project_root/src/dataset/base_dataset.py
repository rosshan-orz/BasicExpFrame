from typing import Dict, Any, Optional, Callable, Union
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from .sample_type import SampleDict

class BaseDataset(Dataset, ABC):
    """

    """
    def __init__(self, file_path: Union[str, Path], transform: Optional[Callable] = None):
        """

        :param file_path:
        :param transform:
        """
        self.file_path = Path(file_path)
        self.transform = transform
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file or directory not found at {self.file_path}")

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.
        :return: total number of samples
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int) -> SampleDict:
        """
        Retrieves a single sample from the dataset at the given index.
        :param index: given index
        :return: a sample from dataset
        """
        raise NotImplementedError