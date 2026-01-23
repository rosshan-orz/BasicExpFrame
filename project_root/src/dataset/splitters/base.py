from abc import ABC, abstractmethod
from typing import Tuple
from torch.utils.data import Dataset

class BaseSplitter(ABC):
    """
    Abstract base class for dataset splitters
    Key interface for splitting dataset to train and validation and test datasets
    """

    @abstractmethod
    def __call__(self, dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Base class for dataset splitters
        :param dataset: torch.utils.data.Dataset, dataset to be splitted
        :return: tuple of train, validation and test datasets
        """
        raise NotImplementedError