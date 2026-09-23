"""Thin Lightning adapter around ordinary PyTorch models."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Optional

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Metric

from src.model import build_criterion, build_metrics, build_optimizer, build_scheduler


class ExperimentLightningModule(pl.LightningModule):
    """Wrap a plain ``nn.Module`` without adding data/business logic."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer_config: Mapping[str, Any],
        scheduler_config: Optional[Mapping[str, Any]] = None,
        metrics: Optional[nn.ModuleDict | Mapping[str, Metric]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer_config = dict(optimizer_config)
        self.scheduler_config = dict(scheduler_config) if scheduler_config else None
        if metrics is None:
            self.val_metrics = nn.ModuleDict()
        elif isinstance(metrics, nn.ModuleDict):
            self.val_metrics = metrics
        else:
            self.val_metrics = nn.ModuleDict(dict(metrics))
        self.test_metrics = nn.ModuleDict(
            {name: copy.deepcopy(metric) for name, metric in self.val_metrics.items()}
        )
        self.save_hyperparameters(ignore=["model", "criterion", "metrics"])

    def forward(self, inputs: Any) -> Mapping[str, torch.Tensor]:
        return self.model(inputs)

    @staticmethod
    def _unpack_batch(batch: Any):
        if isinstance(batch, Mapping):
            return batch["inputs"], batch["targets"]
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        raise TypeError("batches must contain inputs and targets")

    def _shared_step(self, batch: Any, stage: str) -> torch.Tensor:
        inputs, targets = self._unpack_batch(batch)
        outputs = self.model(inputs)
        if not isinstance(outputs, Mapping) or "logits" not in outputs:
            raise TypeError("model forward must return a mapping containing 'logits'")
        loss = self.criterion(outputs["logits"], targets)
        self.log(f"{stage}/loss", loss, on_step=stage == "train", on_epoch=True, prog_bar=True)
        metrics = self.val_metrics if stage == "val" else self.test_metrics
        for name, metric in metrics.items():
            value = metric(outputs["logits"].detach(), targets.detach())
            self.log(f"{stage}/{name}", value, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = build_optimizer(self.optimizer_config, self.parameters())
        if not self.scheduler_config:
            return optimizer
        scheduler = build_scheduler(self.scheduler_config, optimizer)
        scheduler_config: dict[str, Any] = {"scheduler": scheduler}
        if self.scheduler_config.get("name") == "ReduceLROnPlateau":
            scheduler_config["monitor"] = self.scheduler_config.get("monitor", "val/loss")
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
