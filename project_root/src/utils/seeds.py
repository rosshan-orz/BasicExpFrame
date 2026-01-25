import random
import numpy as np
import torch
import os

def seed_setup(seed: int) -> None:
    """
    Fixes random seeds for reproducibility across various libraries.
    :param seed: seed fixed
    :return: nothing
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed fixed to {seed} for random, numpy, torch and CUDA")

def work_init_fn(worker_id: int) -> None:
    """
    Callable for `DataLoader`'s `worker_init_fn` to ensure different seeds for each worker.
    This helps prevent identical data augmentations across workers when using multiprocessing.
    :param worker_id: id of the worker process
    :return: nothing
    """
    worker_seed = torch.initial_seed() % (2 ** 32 - 1) + worker_id
    seed_setup(worker_seed)
    print(f"Dataloader worker {worker_id} has been seeded with {worker_seed}")