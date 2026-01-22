from abc import ABC, abstractmethod
from typing import tuple
from torch.utils.data import Dataset

class BaseSplitter(ABC):
    """
    Abstract base class for dataset splitters
    """