"""Experiment containers produced by the data planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from torch.utils.data import Dataset


@dataclass(frozen=True)
class Experiment:
    """One isolated train/validation/test dataset arrangement."""

    name: str
    train: Dataset
    valid: Dataset
    test: Dataset
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def subject_metadata(self) -> Dict[str, Any]:
        return self.metadata

    @property
    def train_subjects(self):
        return tuple(self.metadata.get("train_subjects", ()))

    @property
    def valid_subjects(self):
        return tuple(self.metadata.get("valid_subjects", ()))

    @property
    def test_subjects(self):
        return tuple(self.metadata.get("test_subjects", ()))
