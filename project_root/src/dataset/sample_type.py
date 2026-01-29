from typing import Dict, Any, Union
from torch import Tensor

# Define the Dataset Protocol as a TypedDict
SampleDict = Dict[str, Union[Tensor, Dict[str, Tensor], Dict[str, Any]]]