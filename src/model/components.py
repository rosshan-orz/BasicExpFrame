"""Configuration-driven model and training component builders."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch
from torch import nn, optim
from torchmetrics import Metric
from torchmetrics.classification import MulticlassAccuracy

from src.core.registry import Registry

from .registry import MODEL_REGISTRY

CRITERION_REGISTRY = Registry("criterion")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
METRIC_REGISTRY = Registry("metric")


@MODEL_REGISTRY.register("MLP")
class MLP(nn.Module):
    """Minimal classifier useful for smoke tests and examples."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_classes < 1:
            raise ValueError("input_dim, hidden_dim, and num_classes must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: Any) -> dict[str, torch.Tensor]:
        if isinstance(inputs, Mapping):
            inputs = inputs["inputs"]
        return {"logits": self.network(inputs.float())}


# A descriptive alias is useful in YAML while keeping one implementation.
MODEL_REGISTRY.register("SimpleMLP")(MLP)


@METRIC_REGISTRY.register("Accuracy")
class Accuracy(MulticlassAccuracy):
    def __init__(self, num_classes: int, **kwargs: Any) -> None:
        super().__init__(num_classes=num_classes, **kwargs)


for _name in ("CrossEntropyLoss", "MSELoss", "L1Loss", "BCEWithLogitsLoss"):
    CRITERION_REGISTRY.register(_name)(getattr(nn, _name))

for _name in ("SGD", "Adam", "AdamW", "RMSprop"):
    OPTIMIZER_REGISTRY.register(_name)(getattr(optim, _name))

for _name in ("StepLR", "MultiStepLR", "CosineAnnealingLR", "ReduceLROnPlateau"):
    SCHEDULER_REGISTRY.register(_name)(getattr(optim.lr_scheduler, _name))


def _build(registry: Registry, config: Any, **kwargs: Any) -> Any:
    if config is None:
        return None
    return registry.build(config, **kwargs)


def build_model(config: Any) -> nn.Module:
    return MODEL_REGISTRY.build(config)


def build_criterion(config: Any) -> nn.Module:
    return CRITERION_REGISTRY.build(config)


def build_optimizer(config: Any, parameters: Iterable[nn.Parameter]) -> optim.Optimizer:
    return OPTIMIZER_REGISTRY.build(config, params=parameters)


def build_scheduler(config: Any, optimizer: optim.Optimizer) -> Any:
    return _build(SCHEDULER_REGISTRY, config, optimizer=optimizer)


def build_metric(config: Any) -> Metric:
    return METRIC_REGISTRY.build(config)


def build_metrics(configs: Optional[Iterable[Any]]) -> nn.ModuleDict:
    metrics = nn.ModuleDict()
    for index, config in enumerate(configs or ()):
        metric = build_metric(config)
        name = config.get("name", str(index)) if isinstance(config, Mapping) else str(index)
        metrics[name] = metric
    return metrics
