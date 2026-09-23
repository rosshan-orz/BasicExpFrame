"""Subject-aware experiment planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

from torch.utils.data import ConcatDataset, Dataset

from .experiment import Experiment
from .npz_dataset import NpzDataset
from .registry import DATASET_REGISTRY, SPLITTER_REGISTRY
from .splitters import BaseSplitter, RandomSplitter


class _EmptyDataset(Dataset):
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int):
        raise IndexError(index)


def _concat(datasets: Sequence[Dataset]) -> Dataset:
    if not datasets:
        return _EmptyDataset()
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(list(datasets))


class ExperimentPlanner:
    """Discover subject files and produce independent ``Experiment`` objects.

    ``dataset_factory`` is the preferred extension point.  It receives a
    subject file path and returns a regular PyTorch Dataset.  A registry-backed
    dataset configuration or the built-in ``NpzDataset`` can be used instead.
    """

    SUPPORTED_TYPES = {"subject_dependent", "cross_subject", "leave_one_subject_out"}

    def __init__(
        self,
        root: Optional[str | Path] = None,
        *,
        experiment_type: str = "subject_dependent",
        splitter: Optional[BaseSplitter] = None,
        dataset_factory: Optional[Callable[[Path], Dataset]] = None,
        dataset_cls: Optional[type] = None,
        dataset_config: Optional[Mapping[str, Any]] = None,
        subjects: Optional[Sequence[str]] = None,
        file_pattern: str = "*.npz",
        seed: Optional[int] = None,
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.experiment_type = experiment_type
        if experiment_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"Unknown experiment_type {experiment_type!r}; "
                f"choose from {sorted(self.SUPPORTED_TYPES)}"
            )
        self.splitter = splitter or RandomSplitter(seed=seed)
        self.dataset_factory = dataset_factory
        self.dataset_cls = dataset_cls
        self.dataset_config = dict(dataset_config or {})
        self.subjects = list(subjects) if subjects is not None else None
        self.file_pattern = file_pattern

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ExperimentPlanner":
        data = config.get("data", config)
        splitter_config = data.get("splitter", {}) or {}
        if isinstance(splitter_config, str):
            splitter = SPLITTER_REGISTRY.build(splitter_config)
        else:
            name = splitter_config.get("name", "RandomSplitter")
            params = dict(splitter_config.get("params", {}))
            splitter = SPLITTER_REGISTRY.build({"name": name, "params": params})
        dataset_config = data.get("dataset", {}) or {}
        subjects = data.get("subjects", data.get("test_subjects"))
        return cls(
            data.get("root"),
            experiment_type=data.get("experiment_type", "subject_dependent"),
            splitter=splitter,
            dataset_config=dataset_config,
            subjects=subjects,
            file_pattern=data.get("file_pattern", "*.npz"),
        )

    def discover_subject_files(self) -> Dict[str, Path]:
        if self.root is None:
            raise ValueError("root is required to discover subject files")
        if not self.root.exists():
            raise FileNotFoundError(f"Data root directory not found: {self.root}")
        files = {path.stem: path for path in sorted(self.root.glob(self.file_pattern))}
        if not files:
            raise FileNotFoundError(f"No files matching {self.file_pattern!r} in {self.root}")
        selected = list(files) if self.subjects is None or "all" in self.subjects else self.subjects
        missing = [subject for subject in selected if subject not in files]
        if missing:
            raise FileNotFoundError(f"Subject files not found: {missing}")
        return {subject: files[subject] for subject in selected}

    def _make_dataset(self, subject: str, path: Path) -> Dataset:
        if self.dataset_factory is not None:
            dataset = self.dataset_factory(path)
        else:
            config = dict(self.dataset_config)
            name = config.pop("name", "NpzDataset")
            params = dict(config.pop("params", {}) or {})
            params["file_path"] = path
            dataset = DATASET_REGISTRY.build({"name": name, "params": params})
        setattr(dataset, "subject_id", subject)
        return dataset

    def _subject_datasets(self, files: Mapping[str, Path]) -> Dict[str, Dataset]:
        return {subject: self._make_dataset(subject, path) for subject, path in files.items()}

    def _subject_dependent(self, datasets: Mapping[str, Dataset]) -> Iterator[Experiment]:
        for subject, dataset in datasets.items():
            train, valid, test = self.splitter(dataset)
            yield Experiment(
                name=f"subject_dependent_{subject}",
                train=train,
                valid=valid,
                test=test,
                metadata={
                    "experiment_type": self.experiment_type,
                    "train_subjects": (subject,),
                    "valid_subjects": (subject,),
                    "test_subjects": (subject,),
                    "subject": subject,
                },
            )

    def _cross_subject(self, datasets: Mapping[str, Dataset]) -> Iterator[Experiment]:
        subjects = list(datasets)
        train_ids, valid_ids, test_ids = self.splitter.split_sequence(subjects)
        yield Experiment(
            name="cross_subject",
            train=_concat([datasets[s] for s in train_ids]),
            valid=_concat([datasets[s] for s in valid_ids]),
            test=_concat([datasets[s] for s in test_ids]),
            metadata={
                "experiment_type": self.experiment_type,
                "train_subjects": tuple(train_ids),
                "valid_subjects": tuple(valid_ids),
                "test_subjects": tuple(test_ids),
            },
        )

    def _leave_one_subject_out(self, datasets: Mapping[str, Dataset]) -> Iterator[Experiment]:
        subjects = list(datasets)
        if len(subjects) < 2:
            raise ValueError("leave_one_subject_out requires at least two subjects")
        for held_out in subjects:
            remaining = [subject for subject in subjects if subject != held_out]
            train_ids, valid_ids, _ = self.splitter.split_sequence(remaining, test_ratio=0.0)
            yield Experiment(
                name=f"leave_one_subject_out_{held_out}",
                train=_concat([datasets[s] for s in train_ids]),
                valid=_concat([datasets[s] for s in valid_ids]),
                test=datasets[held_out],
                metadata={
                    "experiment_type": self.experiment_type,
                    "train_subjects": tuple(train_ids),
                    "valid_subjects": tuple(valid_ids),
                    "test_subjects": (held_out,),
                    "held_out_subject": held_out,
                },
            )

    def plan(self) -> List[Experiment]:
        files = self.discover_subject_files()
        datasets = self._subject_datasets(files)
        if self.experiment_type == "subject_dependent":
            return list(self._subject_dependent(datasets))
        if self.experiment_type == "cross_subject":
            return list(self._cross_subject(datasets))
        return list(self._leave_one_subject_out(datasets))

    def __iter__(self) -> Iterator[Experiment]:
        yield from self.plan()
