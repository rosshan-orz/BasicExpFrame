# Model API Reference

本指南详细说明了 BasicExpFrame 中模型架构相关的 API 签名。

与 Dataset 和 Metric 不同，所有的自定义模型并不继承框架自有的特定基类，而是**必须直接继承 PyTorch 原生的 `torch.nn.Module`**，以保持模型构建的最大灵活性与社区兼容性。

---

## `Module` (Registered)

```python
class torch.nn.Module(*args, **kwargs)
```

作为网络权重的容器。将模型注册到框架中供配置文件动态调用，需在此类顶部添加 `@MODEL_REGISTRY.register()` 装饰器。

### Return Constraints (返回值约束)

在重写 `forward` 方法时，为了适配统一的 `Trainer` 推理和下游 `BaseMetric` 的解析，框架对返回值有如下约束规范：

#### `forward(*args, **kwargs) -> Dict[str, Tensor]`

*(推荐规范)* 执行网络的前向计算过程，并显式返回字典。

* **Returns:**
  包含主要预测结果和（可选的）中间特征的字典。
    * **`logits`** (`Tensor`) – 网络的最终输出结果（必须）。
    * 此字典的其余键（如 `"features"`, `"attention_maps"` 等）可自由定义，后续的 Metric 或 Logger 接口可以直接通过键名捕获这些中间变量。
* **Return type:** `Dict[str, Tensor]`

#### 兼容性说明

若未按照上述规范返回字典，框架内的 `Trainer` 会尝试进行以下向下兼容转换以防止崩溃（**不推荐依赖此特性**）：

* 若返回一阶或多阶**纯张量 (`Tensor`)**：会被自动包裹为字典 `{'logits': outputs}`。
* 若返回**元组或列表 (`Tuple/List`)**：默认取最后一个元素作为主判决输出，即 `{'logits': outputs[-1], 'features': outputs[0]}`。

### Registration Example

```python
# src/models/my_model.py
import torch.nn as nn
from src.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register(name="my_custom_cnn")
class MyCustomCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3)
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        out = self.fc(feat)
        # 兼容基准字典输出规范
        return {"logits": out, "feat": feat}
```
