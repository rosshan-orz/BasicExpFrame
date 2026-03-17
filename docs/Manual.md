# BasicExpFrame 详细使用说明书

## 第一章：项目总述

BasicExpFrame 是一个基于 PyTorch 的通用且模块化的深度学习实验框架。该框架旨在简化学术研究和算法开发中的实验流程，通过配置驱动的设计，使用户能够快速切换模型、数据集、损失函数和优化算法，同时保持代码的整洁与可伸缩性。

### 核心特性

本框架主要具备以下几个核心特性：

- **高度模块化**：通过引入 `Registry`（注册表）机制，彻底解耦了模型（Models）、数据集（Datasets）、评估指标（Metrics）和优化组件。开发者可以方便地横向扩展新模块。
- **配置驱动 (Configuration-driven)**：所有的实验超参数、数据路径以及模型架构选择，均通过统一的 YAML 配置文件进行管理，实现代码与配置分离，便于消融实验和结果复现。
- **自动化训练引擎 (Trainer)**：内置了通用的训练引擎，接管了繁琐的训练循环，并原生支持自动混合精度（AMP）、梯度裁剪、断点续训以及最佳模型验证与保存机制。
- **清晰的实验追踪**：无缝集成 Python logging 与 TensorBoard，自动记录控制台日志并支持多维度的训练指标可视化。
- **适用的特殊领域处理**：内置对 EEG 等时序医学数据的良好支持，如按受试者划分（Within-subject / Cross-subject）的交叉验证策略。

简单来说，BasicExpFrame 让你只需编写核心的“模型定义”与“数据处理”类，并在 YAML 中稍作配置，即可一键启动标准化、工程化的高质量深度学习训练。

## 第二章：核心架构与运行机制

本章简述框架的底层运行机制及核心模块的职责边界。

### 2.1 Registry (注册表) 机制

BasicExpFrame 采用 `src.utils.registry.Registry` 实现对象的动态实例化，以支持配置驱动和模块解耦。其工作流程如下：

- 框架内置多组全局注册表实例，如 `DATASET_REGISTRY`, `MODEL_REGISTRY`, `METRIC_REGISTRY` 等。
- 开发者在自定义类或函数上方添加 `@XXX_REGISTRY.register()` 装饰器进行注册。
- 框架初始化时，通过解析 YAML 配置文件中的 `name` 字段匹配注册表中的类，并将 `params` 字段作为关键字参数完成实例化。

此机制确保了在新增算法、数据集或评估指标时，无需在框架的实例化代码中引入硬编码修改。

### 2.2 Trainer 训练引擎

`src.trainer.trainer.Trainer` 负责执行标准的模型训练及评估流程。其类实例封装了模型、数据加载器、损失函数、优化器及日志记录器等组件。

单次 `trainer.run()` 的工作流如下：

1. **初始化配置**：加载模型及数据至指定计算设备 (CPU/GPU)，读取并配置自动混合精度 (AMP) 状态。若指定 `resume_from`，则加载断点文件恢复训练状态。
2. **Epoch 迭代**：
   - **`train_epoch`**：迭代 `train_loader`，执行前向传播、计算损失与反向传播。执行梯度更新，支持梯度裁剪与 AMP 加速配置。同步调用统一的 `MetricManager` 记录训练指标。
   - **`validate_epoch`**：迭代 `valid_loader`，在 `torch.no_grad()` 下执行前向传播，计算验证集损失及其他评估指标。
   - **学习率调度**：根据调度策略（如按 Epoch 下降，或根据验证集性能 `ReduceLROnPlateau`）更新学习率。
   - **检查点保存**：对比当前 Epoch 监控指标（如 `val/loss`）与历史最佳记录，若当前指标更优，则更新最佳模型权重。默认同步保存每个 Epoch 结束时的最新状态。
3. **测试阶段**：若配置了 `test_loader`，在所有训练 Epoch 结束后触发 `test_epoch`，评估并记录最终测试集的性能指标。

### 2.3 核心模块概览

框架被划分为四个职责隔离的独立模块：

- **Dataset (数据)**：负责读取原始数据、进行特征提取和基础预处理。结合 Dataset Builder 与 Splitter（划分器），支持复杂的交叉验证策略（如跨受试者验证）。👉 [详见 Dataset API 规范](./api/dataset.md)
- **Model (模型)**：负责定义网络架构及前向传播逻辑 `forward()`。规范要求返回统一格式的字典（如包含 `logits` 等键），以兼容各类损失计算接口。👉 [详见 Model API 规范](./api/models.md)
- **Metrics (评估)**：负责计算评价指标。基于面向对象设计规范，开发者提供继承基类并实现 `update` 与 `compute` 方法。所有评价过程均由 `MetricManager` 统一调度并记录。👉 [详见 Metrics API 规范](./api/metrics.md)
- **Trainer (训练)**：负责调度整体软硬件资源执行训练与测试循环（见 2.2 节）。其行为主要通过修改 YAML 配置文件中的超参数控制，通常无需二次开发。👉 [详见 Trainer API 规范](./api/trainer.md)

## 第三章：项目目录结构

良好的目录结构是二次开发的基础。BasicExpFrame 采用标准化的工程目录规范：

```text
BasicExpFrame/
├── config/                  # 配置目录
│   └── template.yaml        # 默认核心配置文件，存放模型、数据、训练器的各类超参数
├── docs/                    # 文档目录
│   ├── api/                 # 各个独立模块的 API 详细说明手册
│   ├── FAQ.md               # 常见问题与排错指南
│   └── Manual.md            # 本手册
├── project_root/            # 核心源码目录
│   ├── main.py              # 训练主入口脚本
│   └── src/                 # 按功能划分的代码包
│       ├── dataset/         # 数据处理模块 (BaseDataset, Splitter, Builder 等)
│       ├── metrics/         # 评价指标模块 (MetricsManager 及各业务指标实现)
│       ├── models/          # 神经网络模型实现
│       ├── trainer/         # 训练周期调度引擎
│       └── utils/           # 通用工具类 (Registry注册表, Checkpoint存档, Logger监控等)
└── README.md                # 项目简介与极简启动说明
```

## 第四章：基础使用指南

框架通过剥离代码逻辑与实验参数，使得启动与复现一次实验非常标准化。

### 4.1 参数配置 (Configuration)

配置项统一管理在 `config/template.yaml` 中，或执行时动态指定的其他 YAML 文件。其生命周期为：
1. 框架入口点 `main.py` 通过命令行读取参数指定的 YAML 文件。
2. 将配置文件解析为支持嵌套对象访问的 `Box` 对象。
3. 框架根据配置结构，通过 Registry 动态组装整个实验图。

### 4.2 启动训练 (Start Training)

标准情况下，用户只需设定好 `config` 后，在根目录直接执行入口脚本：

```bash
python project_root/main.py --config config/template.yaml
```

*注：可通过命令行覆盖部分基础参数，详细参数覆盖规则参考 `main.py` 中的 argparse 配置。*

## 第五章：进阶开发规范

框架设计鼓励开发者通过**继承与注册**的方式横向扩展功能，而不是纵向修改框架源码。

### 标准模块开发范式

以新增一个名为 `MyNewModel` 的深度学习模型为例，标准的开发工作流应遵循以下三步：

1. **新建模块与继承**：
   在 `src/models/` 目录下新建 python 文件，导入并继承目标抽象层/基类（如 `torch.nn.Module`）。

2. **打上注册表标签**：
   引入对应的 Registry（如 `MODEL_REGISTRY`），并在类定义上方添加装饰器：
   ```python
   from src.utils.registry import MODEL_REGISTRY
   import torch.nn as nn

   @MODEL_REGISTRY.register(name="my_new_model")
   class MyNewModel(nn.Module):
       def __init__(self, hidden_dim):
           super().__init__()
           # ...

       def forward(self, x):
           # 规范：返回字典结构
           return {"logits": x}
   ```

5. **配置引用**：
   在 YAML 配置文件中通过注册的 `name` 激活对应模块。以 Model 为例：
   ```yaml
   model:
     name: "my_new_model"
     params:
       hidden_dim: 256
   ```

### 5.2 组件职责解析与开发指引

以下梳理数据、指标、损失三大核心版块的具体扩展规范：

#### 数据流：Dataset 与 Splitter 的协同与区别

在 BasicExpFrame 中，数据流的构建被拆解为两个正交的维度：**读什么数据 (Dataset)** 与 **怎么切分评估 (Splitter & Strategy)**。

- **Dataset (数据集)**
  - **职责**：仅负责从磁盘（通常指单个物理文件，如单一受试者的 `.npz`）中读取数据并封装为 Tensor。它是纯粹的单个数据源抽象。
  - **开发规范**：继承 `src.dataset.base_dataset.BaseDataset`。必须实现 `__len__` 和 `__getitem__`。需要在类上添加 `@DATASET_REGISTRY.register()`。
  - **注意**：`__getitem__` 返回值规范上通常是一个包含 `inputs` 和 `targets` 等键的字典（即 `SampleDict`）。

- **Splitter (划分器)**
  - **职责**：负责接收一个完整的 Dataset（或拼接后的 Dataset），并返回拆分好的三个子集 `(train_dataset, valid_dataset, test_dataset)`。比如最常见的 `RandomSplitter` 就是按比例随机切分。
  - **开发规范**：继承 `src.dataset.splitters.base.BaseSplitter`，实现 `__call__(self, dataset: Dataset)` 方法，并打上 `@SPLITTER_REGISTRY.register()` 标签。

- **`experiment_type` (数据组合与实验策略)**
  - **职责这不是一个类，而是 `DataBuilder` (位于 `src/dataset/builder.py`) 中的分发逻辑。它决定了不同 Dataset 文件如何被组织起来喂给 Splitter。**目前内置了 `subject_dependent` (单受试者内部划分), `cross_subject` (多受试者混合后划分), 以及 `leave_one_subject_out` (留一法验证)。
  - **扩展方法**：若需新增一种由多文件联合的跨域评估新策略，需在此文件下新增一个静态方法（如 `_run_my_strategy`）并接入底部的策略分发字典 `strategies` 中。

#### 评估流：Metric (评价指标)

- **职责**：负责在训练与测试环节，根据模型的输出（如 Logits）与真实标签计算准确率、F1 等指标。
- **开发规范**：
  1. 继承 `src.metrics.abstract.BaseMetric`。
  2. 实现三个核心方法：
     - `reset()`: 清空统计量（在每个 epoch 始末调用）。
     - `update(self, outputs, batch)`: 接收单个 batch 的预测值和真实值，累加统计量。
     - `compute(self)`: 根据统计量算出最终得分。**强制要求返回一个 `Dict[str, float]`**（例如 `{"acc": 0.95}`），这对于日志的自动记录至关重要。
  3. 使用 `@METRIC_REGISTRY.register()` 进行注册。

#### 损失函数：Criterion

- **职责**：负责计算前向传播的 Loss 以供梯度回传。
- **开发规范**：通常情况下，PyTorch 原生的 `nn.CrossEntropyLoss` 已经通过 `CRITERION_REGISTRY` 注册。若需自定义损失函数：
  1. 新建类继承 `torch.nn.Module`。
  2. 实现 `forward(self, logits, targets)` 计算标量 Loss 并返回。
  3. 在类上方加入 `@CRITERION_REGISTRY.register(name="my_loss")`。

掌握上述模块的独立组装与协同机制，即可在不触碰核心训练循环的前提下完成任何算法的平滑迭代。

