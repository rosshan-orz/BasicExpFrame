from ..utils.registry import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

# Register common PyTorch optimizers
OPTIMIZER_REGISTRY.register()(optim.Adam)
OPTIMIZER_REGISTRY.register()(optim.AdamW)
OPTIMIZER_REGISTRY.register()(optim.SGD)
OPTIMIZER_REGISTRY.register()(optim.RMSprop)
OPTIMIZER_REGISTRY.register()(optim.Adagrad)

# You can register custom optimizers here as well if needed
# Example:
# from .my_optimizer import MyCustomOptimizer
# OPTIMIZER_REGISTRY.register()(MyCustomOptimizer)


# Initialize the registry for learning rate schedulers


# Register common PyTorch learning rate schedulers
# Note: Schedulers often require the optimizer as an argument, which will be handled in build.
SCHEDULER_REGISTRY.register()(optim.lr_scheduler.StepLR)
SCHEDULER_REGISTRY.register()(optim.lr_scheduler.MultiStepLR)
SCHEDULER_REGISTRY.register()(optim.lr_scheduler.ExponentialLR)
SCHEDULER_REGISTRY.register()(optim.lr_scheduler.CosineAnnealingLR)
SCHEDULER_REGISTRY.register()(optim.lr_scheduler.ReduceLROnPlateau) # This one is special, might need different handling
SCHEDULER_REGISTRY.register()(optim.lr_scheduler.CyclicLR)
