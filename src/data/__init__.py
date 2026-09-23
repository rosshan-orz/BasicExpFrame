from .base_dataset import BaseDataset
from .experiment import Experiment
from .loaders import build_dataloader, build_loaders
from .npz_dataset import NpzDataset
from .planner import ExperimentPlanner
from .sample_type import SampleDict
from .splitters import BaseSplitter, RandomSplitter, SequentialSplitter

from .registry import DATASET_REGISTRY, SPLITTER_REGISTRY

# Register built-in splitters on package import.
DATASET_REGISTRY.get("NpzDataset")
SPLITTER_REGISTRY.register("RandomSplitter")(RandomSplitter)
SPLITTER_REGISTRY.register("SequentialSplitter")(SequentialSplitter)

__all__ = [
    "BaseDataset",
    "Experiment",
    "ExperimentPlanner",
    "SampleDict",
    "NpzDataset",
    "BaseSplitter",
    "RandomSplitter",
    "SequentialSplitter",
    "DATASET_REGISTRY",
    "SPLITTER_REGISTRY",
    "build_dataloader",
    "build_loaders",
]
