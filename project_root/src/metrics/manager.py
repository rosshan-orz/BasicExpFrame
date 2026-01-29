from typing import Dict, List, Union, Any
from torch import Tensor
from ..utils.registry import METRIC_REGISTRY
from .abstract import BaseMetric

class MetricManager:
    """
    Manage a collection of metrics for reporting performance.
    """
    def __init__(self, metric_config: List[Union[str, Dict[str, Any]]]):
        """

        :param metric_config:
        """
        self.metrics: Dict[str, BaseMetric] = {}
        for config in metric_config:
            metric: BaseMetric = METRIC_REGISTRY.build(config)
            if metric.name in self.metrics:
                raise ValueError(f"Metric {metric.name} already exists")
            self.metrics[metric.name] = metric
        self.reset()

    def reset(self) -> None:
        """
        Reset all managed metrics.
        :return:
        """
        for metric in self.metrics.values():
            metric.reset()

    def update(self, outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> None:
        """
        Update all managed metrics with new predictions and truth.
        :param outputs:
        :param batch:
        :return:
        """
        for metric in self.metrics.values():
            metric.update(outputs, batch)

    def compute(self) -> Dict[str, float]:
        """
        Compute all managed metrics and return their final values.
        :return:
        """
        results = {}
        for metric in self.metrics.values():
            computed_metrics = metric.compute()
            for key, value in computed_metrics.items():
                if key in results:
                    raise ValueError(f"Duplicate metric key {key} found across different metrics")
                results[key] = value
        return results