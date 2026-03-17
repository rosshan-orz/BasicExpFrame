# BasicExpFrame

BasicExpFrame 是一个基于 PyTorch 的通用且模块化的深度学习实验框架。该框架旨在简化实验流程，通过配置驱动（Configuration-driven）的设计，使用户能够快速切换模型、数据集、损失函数和优化算法，同时保持代码的整洁与可扩展性。

## 核心特性

- **高度模块化**：通过 `Registry` 机制解耦模型、数据集、指标和优化器，方便扩展。
- **配置驱动**：所有实验参数（包括超参数、数据路径、模型选择等）均通过 YAML 文件统一管理。
- **自动化流程**：内置通用的 `Trainer`，支持自动混合精度（AMP）、梯度裁剪、断点续训以及最佳模型保存。
- **实验追踪**：集成日志记录与指标管理，支持多维度评估模型性能。
- **EEG 领域优化**：内置对 EEG 数据处理（如按受试者划分）的良好支持。

## 项目结构

```text
BasicExpFrame/
├── project_root/
│   ├── main.py                 # 实验启动入口
│   ├── config/                 # 配置文件目录
│   │   └── template.yaml       # 实验配置模板
│   └── src/                    # 核心源代码
│       ├── dataset/            # 数据加载与划分逻辑
│       ├── metrics/            # 指标计算与管理
│       ├── models/             # 模型定义
│       ├── trainer/            # 训练引擎与损失函数
│       └── utils/              # 注册表、日志、检查点等工具类
├── requirements.txt            # 项目依赖
└── README.md                   # 项目说明
```

## 快速开始

### 1. 安装依赖
确保已安装 Python 3.8+ 和 PyTorch，然后运行：
```bash
pip install -r requirements.txt
```

### 2. 准备数据
按照 `project_root/config/template.yaml` 中的 `data.root` 配置，将你的 `.npz` 或相关数据文件放入指定目录。

### 3. 配置实验
参考 `project_root/config/template.yaml` 创建你自己的配置文件。你可以修改：
- `model`: 选择注册的模型（如 `EEGNet`）。
- `data`: 设置数据路径及划分策略（如 `within_subject`）。
- `optimizer`/`scheduler`: 配置学习率及优化算法。
- `trainer`: 设置 Epoch 数量、是否使用 AMP 等。

### 4. 运行实验
使用以下命令启动实验：
```bash
python project_root/main.py --config project_root/config/your_config.yaml
```

## 开发指南

### 添加新模型
1. 在 `project_root/src/models/` 下创建你的模型文件。
2. 使用 `@MODEL_REGISTRY.register()` 装饰器注册你的模型类。
3. 在 `project_root/src/models/__init__.py` 中导入该类。

### 添加新指标
1. 在 `project_root/src/metrics/` 中实现指标逻辑。
2. 使用 `@METRIC_REGISTRY.register()` 进行注册。

## 贡献
欢迎提交 Issue 或 Pull Request 来改进此框架。

## 许可证
[MIT License](LICENSE)
