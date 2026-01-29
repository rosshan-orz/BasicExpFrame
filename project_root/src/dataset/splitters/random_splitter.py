from typing import Tuple, Optional
from torch.utils.data import Dataset, Subset
from .base import BaseSplitter
import random

class RandomSplitter(BaseSplitter):
    """
    Split dataset into sets based on random seed shuffle and split ratio
    """
    def __init__(self, train_ratio: float = 0.75, valid_ratio: float = 0.125, test_ratio: float = 0.125, seed: Optional[int] = None):
        """
        Initialize random splitter
        :param train_ratio: ratio of training data
        :param valid_ratio: ratio of validation data
        :param test_ratio: ratio of testing data
        :param seed: random seed
        """
        if not(0 <= train_ratio <= 1 and 0 <= valid_ratio <= 1 and 0 <= test_ratio <= 1 and abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-9):
            raise ValueError("train_ratio, valid_ratio, and test_ratio should be between 0 and 1. Sum of them should be equal to 1")
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Split dataset into sets based on random seed shuffling
        :param dataset:
        :return:
        """
        dataset_len = len(dataset)
        indices = list(range(dataset_len))

        if self.seed is not None:
            random.seed(self.seed)
        random.shuffle(indices)

        valid_len = int(dataset_len * self.valid_ratio)
        test_len = int(dataset_len * self.test_ratio)
        train_len = dataset_len - valid_len - test_len

        train_indices = indices[:train_len]
        valid_indices = indices[train_len:train_len + valid_len]
        test_indices = indices[train_len + valid_len:]

        return Subset(dataset, train_indices), Subset(dataset, valid_indices), Subset(dataset, test_indices)
