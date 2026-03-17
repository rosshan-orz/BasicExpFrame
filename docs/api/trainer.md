# Trainer API Reference

本指南梳理了 `Trainer` 类的核心签名及其对外暴露的重写接口。虽然在基础开发中很少需要强制继承重写 `Trainer`，但理解其架构对于定位 Bug 与微调控制逻辑极具价值。

---

## `Trainer`

```python
class src.trainer.trainer.Trainer(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader, criterion: torch.nn.Module, config: Box, logger: BaseLogger, metrics_manager: MetricManager, device: torch.device)
```

整体实验的统合管理器。在 `main.py` 流水线中，它接管由 DataBuilder 和 Registry 构建的外围对象序列，并拉起完整的深度学习生命周期。

### Parameters

* **model** (`torch.nn.Module`) – 待训练的网络模型实例。
* **optimizer** (`torch.optim.Optimizer`) – 模型参数优化器。
* **scheduler** (`Any`, optional) – 学习率调度器（如 `ReduceLROnPlateau` 等）。
* **train_loader** (`DataLoader`) – 训练集加载器。
* **valid_loader** (`DataLoader`) – 验证集加载器。
* **test_loader** (`DataLoader`) – 测试集加载器。
* **criterion** (`torch.nn.Module`) – 损失函数的实例化评估器。
* **config** (`Box`) – 全局级配置解析字典。
* **logger** (`BaseLogger`) – 日志及实验追踪器实例。
* **metrics_manager** (`MetricManager`) – 统一代理底层各 `BaseMetric` 接口的聚合采集器。
* **device** (`torch.device`) – 软硬件挂载点配置 (如 `cuda:0` 或 `cpu`)。

### Workflow Methods (核心重写接口)

对于对抗生成（GAN）或自监督学习等需接管梯度操作的高阶控制工作，可通过继承重写以下核心单步流。

#### `_forward(inputs: Any) -> Dict[str, Tensor]`

此阶段完成底层模型推演 `model(inputs)`。
* **Parameters:** **inputs** (`Any`) – 从当前 batch 解析并推送至 GPU/CPU 侧的特征集合。若输入为字典则利用 `**inputs` 解包喂给模型。
* **Returns:** 确保被包装成了安全的字典结构再向下流转给 Metric 统合器。
* **Return type:** `Dict[str, Tensor]`

#### `train_step(batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]`

执行最小粒度的核心训练单步：抽取数据、前向传播、求导并更新梯度。
内置了针对 AMP (自动混合精度 `scaler`) 以及梯度裁剪 (`grad_clip_value`) 的护航代码。
* **Parameters:** **batch** (`Dict[str, Any]`) – 当前正在处理的独立批次数据。
* **Returns:** 第一项为已无计算图附着的独立 loss 标量值，第二项为保留供 Metric 取样的模型预测原格式输出。
* **Return type:** `Tuple[Tensor, Dict[str, Tensor]]`

#### `_evaluation_step(batch: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Tensor]]`

与 `train_step` 结构一致但意图正交。在通过 `torch.no_grad()` 上下文关闭任何梯度流向干涉与 Dropout 层干预后，专门进行高纯度的验证前向。

---

### Configuration Driven

如不愿通过原生类重写拓展代码，可以通过修改对应 `config/template.yaml` 根下级的 `trainer:` 命名空间，控制默认行为。

* **`epochs`** (`int`) – 外部强制设定的运行迭代大周期。
* **`monitor`** (`str`, optional) – 重大性能落盘和调度判定的核心关注指标键，默认为 `"val/loss"`。
* **`monitor_mode`** (`str`, optional) – 该关注指标的优化方向，判定更优越是 `"min"` 还是 `"max"`。
* **`save_dir`** (`str`) – 指向历史 Checkpoint 落盘目录。
* **`use_amp`** (`bool`, optional) – 开启半精度 FP16 内存推流。
* **`grad_clip`** (`float`, optional) – 对梯度的绝对范数裁剪值。
* **`resume_from`** (`str`, optional) – 指定可续接训练断点的权重文件相对位置。
