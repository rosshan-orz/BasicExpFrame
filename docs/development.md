# 开发与扩展

## 分层约定

- `src/core` 不依赖业务代码或 Lightning。
- `src/data` 只实现 Dataset、划分和实验组织，不导入 Lightning。
- `src/model` 只定义普通 `torch.nn.Module` 和组件构造，不持有 optimizer、scheduler 或 metrics 实例。
- `src/trainer` 是唯一 Lightning 适配边界。
- `project/main.py` 负责编排，不实现具体数据读取或网络结构。

## 添加 Dataset

继承 `BaseDataset`，实现 `__len__` 和 `__getitem__`。样本建议返回 `{"inputs": ..., "targets": ...}`，其中 `inputs` 和 `targets` 应为可由 PyTorch 默认 collate 处理的值。自定义文件格式可在 planner 的 `dataset` 配置指向注册类。

```python
from src.data import BaseDataset, DATASET_REGISTRY

@DATASET_REGISTRY.register("MyDataset")
class MyDataset(BaseDataset):
    def __init__(self, file_path, transform=None):
        super().__init__(file_path, transform)
        # 读取并缓存文件中的样本

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        return {"inputs": sample[0], "targets": sample[1]}
```

导入定义该类的模块后注册才会生效。配置 `data.dataset.name: MyDataset`，Planner 会向构造参数注入发现到的 `file_path`。

## 添加 Splitter

继承 `BaseSplitter`，实现 `__call__(dataset)` 和 `split_indices(size, test_ratio=None)`。后一接口也用于按 subject 列表切分，因此必须返回三个不重叠、覆盖所有输入索引的列表。然后向 `SPLITTER_REGISTRY` 注册。

## 添加模型

定义普通 `nn.Module`，`forward` 返回含 `logits` 的 mapping，并注册到 `MODEL_REGISTRY`。输出 logits 和 targets 的形状/类型必须符合所选 loss 的要求。

## 添加 loss、optimizer、scheduler 或 metric

对应 Registry 导出自 `src.model`：`CRITERION_REGISTRY`、`OPTIMIZER_REGISTRY`、`SCHEDULER_REGISTRY`、`METRIC_REGISTRY`。用 `@REGISTRY.register("ConfigName")` 注册可构造的类/函数。Optimizer 由框架注入模型参数；Scheduler 由框架注入 optimizer。Metric 使用 torchmetrics 接口，`update/prediction` 输入顺序由具体 torchmetrics 类定义。

## 验证

```bash
uv run pytest tests/
uv run python project/main.py --config config/template.yaml
```

数据策略变更应增加 subject 泄漏测试；Lightning 集成应增加极小 CPU 端到端测试，避免测试依赖 CUDA。
