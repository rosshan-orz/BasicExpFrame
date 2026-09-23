"""Construction of Lightning Trainer instances from plain mappings."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import lightning.pytorch as pl


def create_trainer(
    config: Optional[Mapping[str, Any]] = None,
    *,
    logger: Any = None,
    callbacks: Optional[Sequence[Any]] = None,
    default_root_dir: Optional[str] = None,
) -> pl.Trainer:
    values = dict(config or {})
    if "trainer" in values and isinstance(values["trainer"], Mapping):
        values = dict(values["trainer"])
    aliases = {
        "epochs": "max_epochs",
        "grad_clip": "gradient_clip_val",
    }
    for source, target in aliases.items():
        if source in values and target not in values:
            values[target] = values.pop(source)
    if "use_amp" in values:
        use_amp = values.pop("use_amp")
        if use_amp and "precision" not in values:
            values["precision"] = "16-mixed"
    values.setdefault("enable_progress_bar", False)
    values.setdefault("logger", logger)
    values.setdefault("callbacks", list(callbacks or []))
    if default_root_dir is not None:
        values.setdefault("default_root_dir", default_root_dir)
    return pl.Trainer(**values)
