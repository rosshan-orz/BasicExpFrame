# 配置文件说明 (Configuration Guide)

本项目使用 YAML 格式配置文件（以 `project_root/config/template.yaml` 为模板）来管理所有实验设置。
得益于注册表（Registry）机制，配置文件可以直接将参数映射到对应类的初始化函数上，从而实现灵活解耦。

本文档对配置文件的各个模块进行了详细分类、说明，并提供了对应的 API 和源码链接。

---

## 1. 实验基础设置 (General)

控制实验最基础的全局属性。

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `experiment_name` | `str` | 实验的名称，通常用于构建日志和输出目录名。 |
| `seed` | `int` | 全局随机种子，用于保证实验的可重复性（如 `42`）。 |
| `device` | `str` | 运行设备，支持 `"cuda"` (GPU) 或 `"cpu"`。 |
| `output_dir` | `str` | 所有实验输出的基础保存路径，例如 `"./output"`。 |

---

## 2. 数据配置 (Data)

集中管理数据集加载、划分以及相关超参数，对应于 `src/dataset` 目录下的组件。

- **`data.root`**: `str`。存储受试者数据所在的根目录。
- **`data.file_ext`**: `str`。受试者数据文件的扩展名（如 `"npz"`, `"npy"`, `"mat"`, `"csv"`，默认为 `"npz"`）。
- **`data.experiment_type`**: `str`。数据划分策略（如 `"within_subject"`, `"leave_one_out"`, `"cross_subject_specific"`）。
- **`data.test_subjects`**: `list`。本次实验需要运行的受试者列表配置，`["all"]` 表示运行所有受试者。

### 2.1 数据集 (Dataset)
负责读取和处理数据的 Dataset 类设置。相关的组件注册于 `DATASET_REGISTRY` 注册表中。
- **源码对应**: [`src/dataset/`](../project_root/src/dataset)
- **`data.dataset.name`**: 需要使用的 Dataset 类名（如 `"NpzDataset"`）。
- **`data.dataset.params`**: 传递给所选 Dataset 类 `__init__` 初始化方法的键值对。

### 2.2 数据划分 (Splitter)
负责将数据拆分为训练集、验证集和测试集。使用 `SPLITTER_REGISTRY` 进行注册。
- **源码对应**: [`src/dataset/splitters/`](../project_root/src/dataset/splitters)
- **`data.splitter.name`**: 划分器类名（如 `"SequentialSplitter"`）。
- **`data.splitter.params`**: 划分器的参数（例如 `ratio: 0.8`, `seed: 42`）。

### 2.3 数据加载器 (DataLoader)
控制如何将数据打包为 Batch。
- **API 通向**: 基于 [PyTorch 的 `torch.utils.data.DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)。
- **`data.loader.batch_size`**: `int`。每个 Batch 的样本数量。
- **`data.loader.num_workers`**: `int`。数据加载的子进程数（`0` 表示在主进程加载）。

---

## 3. 模型配置 (Model)

定义所使用的神经网络架构及其超参数。在 `MODEL_REGISTRY` 中注册。
- **源码对应**: [`src/models/`](../project_root/src/models)
- **`model.name`**: 模型类名（如 `"EEGNet"`）。
- **`model.params`**: 对应网络类 `__init__` 函数的定义（例如传递 `num_classes`, `channels` 等）。

---

## 4. 损失函数配置 (Loss Function)

定义优化目标的损失函数，由 `CRITERION_REGISTRY` 注册管理，是对 PyTorch 损失函数的封装。
- **源码对应**: [`src/trainer/criterion.py`](../project_root/src/trainer/criterion.py)
- **API 通向**: 基于 [PyTorch Loss Functions (`torch.nn`)](https://pytorch.org/docs/stable/nn.html#loss-functions)。
- **`loss.name`**: 损失函数的类名（如 `"CrossEntropyLoss"`）。
- **`loss.params`**: 对应损失函数初始化的参数（如 `reduction: "mean"`、类别权重 `weight` 等）。

---

## 5. 优化器配置 (Optimizer)

用于更新模型参数的优化算法。注册于 `OPTIMIZER_REGISTRY`。
- **源码对应**: [`src/trainer/builder.py`](../project_root/src/trainer/builder.py)
- **API 通向**: 基于 [PyTorch Optimizers (`torch.optim`)](https://pytorch.org/docs/stable/optim.html)。
- **`optimizer.name`**: 优化算法类名（如 `"AdamW"`）。
- **`optimizer.params`**: 传给优化器的具体超参数，最常见的包括 `lr` (学习率) 和 `weight_decay`。

---

## 6. 学习率调度器配置 (Scheduler)

(可选配置) 用于在训练过程中动态调整学习率。如果是 `null` 或者此项不存在，则不使用 Scheduler。注册于 `SCHEDULER_REGISTRY`。
- **源码对应**: [`src/trainer/builder.py`](../project_root/src/trainer/builder.py)
- **API 通向**: 基于 [PyTorch LR Schedulers (`torch.optim.lr_scheduler`)](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)。
- **`scheduler.name`**: 调度策略名（如 `"StepLR"`, `"ReduceLROnPlateau"`）。
- **`scheduler.params`**: 初始化此 Scheduler 类的参数配置（如 `step_size`, `gamma`, 等）。

---

## 7. 训练器配置 (Trainer)

此部分的参数直接应用于 `Trainer` 类的相关行为。
- **源码对应**: [`src/trainer/trainer.py` 的 `Trainer` 类](../project_root/src/trainer/trainer.py)

| 参数名 | 类型 | 说明 |
| :--- | :--- | :--- |
| `epochs` | `int` | 训练的总 Epoch 阶段数量。 |
| `use_amp` | `bool` | 是否开启自动混合精度加速 (Automatic Mixed Precision)。 |
| `grad_clip` | `float` \| `null` | 梯度裁剪的最大范数（避免梯度爆炸），`None` 则不裁剪。 |
| `monitor` | `str` | 用作保存最佳模型以及 Early Stopping 的监控指标名（如 `"val/accuracy"`）。 |
| `monitor_mode` | `str` | 监控指标方向，`"max"` 代表越大越好 (例如准确率)，`"min"` 代表越小越好 (例如 Loss)。 |
| `resume_from` | `str` \| `null` | 指定从某个具体的 checkpoint 文件路径 (`.pth`) 恢复训练。 |
| `debug` | `bool` | 调试模式开关；开启后每个阶段只跑 1 个 Batch 快速验证代码流程。 |

---

## 8. 评估指标配置 (Metrics)

定义在验证和测试过程中计算的指标，可配置为列表（即支持同时监控多个指标）。统一在 `METRIC_REGISTRY` 中管理。
- **源码对应**: [`src/metrics/`](../project_root/src/metrics)
- **参数列表**:
  - `name`: 对应的指标类名（如 `"Accuracy"`, `"F1Score"`）。
  - `params`: 传给该指标初始化的参数，如包含 `name: "val/accuracy"` 用以规定其输出在 log/TensorBoard 中的名称。
