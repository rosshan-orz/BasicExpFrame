"""Standard callback factory."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def build_callbacks(config: Optional[Mapping[str, Any]] = None):
    values = dict(config or {})
    callbacks = []
    checkpoint = values.get("checkpoint", {}) or {}
    if checkpoint is not False:
        callbacks.append(ModelCheckpoint(**checkpoint))
    early = values.get("early_stopping")
    if early:
        callbacks.append(EarlyStopping(**early))
    lr_monitor = values.get("learning_rate_monitor")
    if lr_monitor is not False:
        callbacks.append(LearningRateMonitor(**(lr_monitor or {})))
    return callbacks
