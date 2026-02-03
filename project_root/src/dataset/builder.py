from typing import Iterator, Tuple, Dict, Any, Union, List
from pathlib import Path
from box import Box
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torch.utils.data.backward_compatibility import worker_init_fn

from .sample_type import SampleDict
from ..utils.registry import DATASET_REGISTRY, SPLITTER_REGISTRY
from .splitters.base import BaseSplitter

class DataBuilder:
    """

    """

    @staticmethod
    def _build_loaders(loader_config: Box, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Helper to build dataloaders."""
        batch_size = loader_config.batch_size
        num_workers = loader_config.num_workers

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available()
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available()
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available()
        )

        print(f"Dataloaders created with batch size {batch_size}, num workers {num_workers}")
        return train_loader, valid_loader, test_loader

    @staticmethod
    def _split_dataset(dataset: Dataset, data_config, splitter_config, splitter) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """

        :param dataset:
        :param data_config:
        :param splitter_config:
        :param splitter:
        :return:
        """
        print(f"Using split strategy: {splitter_config.name} with params: {splitter_config.get('params', {})}")

        train_dataset, valid_dataset, test_dataset = splitter(dataset)
        print(f"Split dataset: Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")
        train_loader, valid_loader, test_loader = DataBuilder._build_loaders(data_config.loader, train_dataset, valid_dataset, test_dataset)

        return train_loader, valid_loader, test_loader

    @staticmethod
    def _run_subject_dependent(target_subjects: List[str], all_files: List[Path], root_path: Path, data_config: Box, splitter: BaseSplitter) -> Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]:
        splitter_config = data_config.splitter
        for subject_stem in target_subjects:
            subject_file = root_path / f"{subject_stem}.npz"
            if not subject_file.exists():
                raise FileNotFoundError(f"Subject file not found at {subject_file}")
            
            experiment_name = f"subject_dependent_{splitter_config.name}_{subject_stem}"
            print(f"Building Experiment: {experiment_name}")

            dataset_config = data_config.dataset
            dataset_config.params.file_path = str(subject_file)
            full_dataset = DATASET_REGISTRY.build(dataset_config)
            print(f"Create dataset {dataset_config.name} for {subject_stem} with {len(full_dataset)} samples")

            train_loader, valid_loader, test_loader = DataBuilder._split_dataset(full_dataset, data_config, splitter_config, splitter)
            yield experiment_name, train_loader, valid_loader, test_loader

    @staticmethod
    def _run_cross_subject(target_subjects: List[str], all_files: List[Path], root_path: Path, data_config: Box, splitter: BaseSplitter) -> Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]:
        splitter_config = data_config.splitter
        print("Building Cross-Subject Experiment (Combined Dataset)")
        datasets = []
        for subject_stem in target_subjects:
            subject_file = root_path / f"{subject_stem}.npz"
            if not subject_file.exists():
                raise FileNotFoundError(f"Subject file not found at {subject_file}")

            dataset_config = data_config.dataset
            dataset_config.params.file_path = str(subject_file)
            ds = DATASET_REGISTRY.build(dataset_config)
            datasets.append(ds)
        
        experiment_name = f"cross_subject_{splitter_config.name}_{target_subjects}"
        print(f"Building Experiment: {experiment_name}")
        
        full_dataset = ConcatDataset(datasets)
        print(f"Combined {len(datasets)} subjects. Total samples: {len(full_dataset)}")

        train_loader, valid_loader, test_loader = DataBuilder._split_dataset(full_dataset, data_config, splitter_config, splitter)
        yield experiment_name, train_loader, valid_loader, test_loader

    @staticmethod
    def _run_leave_one_subject_out(target_subjects: List[str], all_files: List[Path], root_path: Path, data_config: Box, splitter: BaseSplitter) -> Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]:
        splitter_config = data_config.splitter
        print("Building Leave-One-Subject-Out Experiments")
        
        all_subject_stems = [f.stem for f in all_files]

        for valid_subject_stem in target_subjects:
            experiment_name = f"leave_one_subject_out_{splitter_config.name}_test_on_{valid_subject_stem}"
            print(f"--- Building Experiment: {experiment_name} ---")

            # 1. Build Test Dataset
            valid_subject_file = root_path / f"{valid_subject_stem}.npz"
            dataset_config = data_config.dataset
            dataset_config.params.file_path = str(valid_subject_file)
            valid_dataset = DATASET_REGISTRY.build(dataset_config)
            print(f"Test dataset: {valid_subject_stem} with {len(valid_dataset)} samples")

            # 2. Build Train/Valid Source (All other subjects)
            train_datasets = []
            train_subject_stems = [s for s in all_subject_stems if s != valid_subject_stem]
            
            if not train_subject_stems:
                raise ValueError(f"Cannot perform LOSO with only one subject ({valid_subject_stem}).")

            for train_stem in train_subject_stems:
                train_file = root_path / f"{train_stem}.npz"
                dataset_config.params.file_path = str(train_file)
                ds = DATASET_REGISTRY.build(dataset_config)
                train_datasets.append(ds)
            
            train_valid_source = ConcatDataset(train_datasets)
            
            # 3. Split Train/Valid using the splitter
            train_dataset, _, test_dataset = splitter(train_valid_source)
            
            train_loader, valid_loader, test_loader = DataBuilder._build_loaders(data_config.loader, train_dataset, valid_dataset, test_dataset)
            yield experiment_name, train_loader, valid_loader, test_loader

    @staticmethod
    def build_experiments(config: Box) -> Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]:

        data_config = config.data
        root_path = Path(data_config.root)
        if not root_path.exists():
            raise FileNotFoundError(f"Data root directory not found at {root_path}")

        # Check experiment type from config, default to subject_dependent
        experiment_type = data_config.get("experiment_type", "subject_dependent")
        test_subjects = data_config.get("test_subjects", ["all"])
        splitter_config = data_config.splitter
        splitter: BaseSplitter = SPLITTER_REGISTRY.build(splitter_config)

        all_subject_files: List[Path] = sorted(list(root_path.glob("*.npz")))
        if not all_subject_files:
            raise FileNotFoundError(f"No NPZ files found in {root_path}")
        if "all" in test_subjects:
            actual_test_sub_stems = [file.stem for file in all_subject_files]
        else:
            actual_test_sub_stems = test_subjects

        print(f"Detected subjects files: {[file.name for file in all_subject_files]}")
        print(f"Experiments will run for subjects: {actual_test_sub_stems}")

        # Strategy Dispatcher
        strategies = {
            "subject_dependent": DataBuilder._run_subject_dependent,
            "cross_subject": DataBuilder._run_cross_subject,
            "leave_one_subject_out": DataBuilder._run_leave_one_subject_out,
        }

        strategy_fn = strategies.get(experiment_type)
        if not strategy_fn:
            raise ValueError(f"Unknown experiment_type: '{experiment_type}'. Available: {list(strategies.keys())}")

        yield from strategy_fn(actual_test_sub_stems, all_subject_files, root_path, data_config, splitter)
