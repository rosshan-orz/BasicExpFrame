from typing import Iterator, Tuple, Dict, Any, Union, List
from pathlib import Path
from box import Box
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.backward_compatibility import worker_init_fn

from .sample_type import SampleDict
from ..utils.registry import DATASET_REGISTRY, SPLITTER_REGISTRY
from .splitters.base import BaseSplitter

class DataBuilder:
    """

    """

    @staticmethod
    def build_experiments(config: Box) -> Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]:

        data_config = config.data
        root_path = Path(data_config.root)
        if not root_path.exists():
            raise FileNotFoundError(f"Data root directory not found at {root_path}")

        split_strategy = data_config.split_strategy
        test_subjects = data_config.get("test_subjects", ["all"])

        all_subject_files: List[Path] = sorted(list(root_path.glob("*.npz")))
        if not all_subject_files:
            raise FileNotFoundError(f"No NPZ files found in {root_path}")
        if "all" in test_subjects:
            actual_test_sub_stems = [file.stem for file in all_subject_files]
        else:
            actual_test_sub_stems = test_subjects

        print(f"Detected subjects files: {[file.name for file in all_subject_files]}")
        print(f"Experiments will run for subjects: {actual_test_sub_stems}")

        for subject_stem in actual_test_sub_stems:
            subject_file = root_path / f"{subject_stem}.npz"
            if not subject_file.exists():
                raise FileNotFoundError(f"Subject file not found at {subject_file}")
            experiment_name = f"{split_strategy}_{subject_stem}"
            print(f"Building Experiment: {experiment_name}")

            # build dataset from registry
            dataset_config = data_config.dataset
            dataset_config.params.file_path = str(subject_file)
            full_dataset = DATASET_REGISTRY.build(dataset_config)
            print(f"Create dataset {dataset_config.name} for {subject_stem} with {len(full_dataset)} samples")

            # build splitter
            splitter_config = data_config.splitter
            splitter: BaseSplitter = SPLITTER_REGISTRY.build(splitter_config)
            print(f"Using split strategy: {splitter_config.name} with params: {splitter_config.get('params', {})}")

            train_dataset, valid_dataset, test_dataset = splitter(full_dataset)
            print(f"Split dataset: Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

            # build dataloaders
            loader_config = data_config.loader
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

            yield experiment_name, train_loader, valid_loader, test_loader
