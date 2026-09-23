# Model API

## Registry

`src.model` 导出 `MODEL_REGISTRY`、`CRITERION_REGISTRY`、`OPTIMIZER_REGISTRY`、`SCHEDULER_REGISTRY`、`METRIC_REGISTRY`。

内置模型：`MLP(input_dim, hidden_dim=32, num_classes=2, dropout=0.0)`，也可用别名 `SimpleMLP`。它是普通 `torch.nn.Module`，输入 Tensor 或含 inputs 的 mapping，返回 `{"logits": tensor}`。

内置组件名称：

- Criterion：`CrossEntropyLoss`、`MSELoss`、`L1Loss`、`BCEWithLogitsLoss`
- Optimizer：`SGD`、`Adam`、`AdamW`、`RMSprop`
- Scheduler：`StepLR`、`MultiStepLR`、`CosineAnnealingLR`、`ReduceLROnPlateau`
- Metric：`Accuracy`（包装 torchmetrics `MulticlassAccuracy`，需提供 `num_classes`）

## Builder 函数

- `build_model(config) -> nn.Module`
- `build_criterion(config) -> nn.Module`
- `build_optimizer(config, parameters) -> torch.optim.Optimizer`：自动把参数迭代器以 `params` 参数传给优化器。
- `build_scheduler(config, optimizer)`：把 optimizer 注入调度器。
- `build_metric(config) -> torchmetrics.Metric`
- `build_metrics(configs) -> nn.ModuleDict`

所有 config 使用 `{name: ..., params: {...}}` 形式。扩展自定义模型时，forward 输出必须至少有 `logits` 键；扩展 Metric 时应使用 torchmetrics Metric API。
