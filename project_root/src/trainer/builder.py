from typing import Optional, Any
import torch
from box import Box

from ..utils.registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY

def build_optimizer(config: Box, params_to_optimize) -> torch.optim.Optimizer:
    """
    Builds an optimizer from config.
    :param config: Configuration box containing optimizer settings
    :param params_to_optimize: Model parameters to optimize
    :return: Initialized Optimizer
    """
    optimizer_config = config.optimizer
    if not isinstance(optimizer_config, Box):  # Handle case where optimizer is just a string name
        optimizer_name = optimizer_config
        optimizer_params = {}
    else:
        optimizer_name = optimizer_config.name
        optimizer_params = optimizer_config.get("params", {})

    # Optimizer expects 'params' as the first argument, so we handle it specially
    return OPTIMIZER_REGISTRY.get(optimizer_name)(params_to_optimize, **optimizer_params)


def build_scheduler(config: Box, optimizer: torch.optim.Optimizer) -> Optional[Any]:
    """
    Builds a scheduler from config.
    :param config: Configuration box containing scheduler settings
    :param optimizer: The optimizer instance associated with the scheduler
    :return: Initialized Scheduler or None
    """
    scheduler_config = config.get("scheduler", None)
    if scheduler_config is None:
        return None

    if not isinstance(scheduler_config, Box):
        scheduler_name = scheduler_config
        scheduler_params = {}
    else:
        scheduler_name = scheduler_config.name
        scheduler_params = scheduler_config.get("params", {})

    # Schedulers expect the optimizer instance as the first argument
    return SCHEDULER_REGISTRY.get(scheduler_name)(optimizer, **scheduler_params)