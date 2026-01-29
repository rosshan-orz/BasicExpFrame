import torch.nn as nn

from ..utils.registry import CRITERION_REGISTRY

CRITERION_REGISTRY.register()(nn.CrossEntropyLoss)
CRITERION_REGISTRY.register()(nn.MSELoss)
CRITERION_REGISTRY.register()(nn.L1Loss)
CRITERION_REGISTRY.register()(nn.BCEWithLogitsLoss)
CRITERION_REGISTRY.register()(nn.NLLLoss)
CRITERION_REGISTRY.register()(nn.KLDivLoss)

# Example of a custom criterion
# @CRITERION_REGISTRY.register()
# class CustomWeightedLoss(nn.Module):
#     def __init__(self, weight_param: float = 1.0):
#         super().__init__()
#         self.weight_param = weight_param
#         self.mse = nn.MSELoss()
#
#     def forward(self, logits, targets):
#         # Custom logic here
#         return self.mse(logits, targets) * self.weight_param
