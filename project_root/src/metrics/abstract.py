from abc import ABC, abstractmethod
from typing import Dict, Any
from torch import Tensor

class BaseMetric(ABC):
    """
    Abstract base class for all metrics. Define the interface for metrics.
    """
    def __init__(self, name: str):
        """

        :param name:
        """
        self.name = name

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the internal state of the metric, clearing any accumulated statistics.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> None:
        """
        Update the internal state of the metric with new predictions and truth.
        :param outputs:
        :param batch:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute the final metric values from the accumulated statistics.
        :return: A dictionary where keys are metric names and values are their corresponding values.
        """
        raise NotImplementedError