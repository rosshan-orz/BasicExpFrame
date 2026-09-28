# BasicExpFrame

基于 YAML、PyTorch 和 PyTorch Lightning 的实验框架。Dataset、Splitter 和模型保持普通 PyTorch 实现；Lightning 只负责训练执行。支持 subject-dependent、cross-subject 和 leave-one-subject-out（LOSO）实验。

## 快速开始

```bash
python project/main.py --config config/template.yaml
```

当前 `main` 分支只保留运行代码、配置模板和文档；Python/uv 环境文件与测试保留在开发分支中。因此运行前请在目标环境自行安装项目依赖。`config/template.yaml` 是不依赖实验数据的 smoke 配置；要训练模型，请参考[使用指南](docs/usage.md)创建数据配置。

## 文档

- [使用指南](docs/usage.md)：安装、数据格式、运行实验、CUDA、输出结构
- [配置参考](docs/configuration.md)：完整 YAML 字段说明与示例
- [开发与扩展](docs/development.md)：添加 Dataset、Splitter、模型与组件
- API： [Core](docs/api/core.md) · [Data](docs/api/data.md) · [Model](docs/api/model.md) · [Trainer](docs/api/trainer.md)

## 目录职责

```text
config/       YAML 配置文件
project/      入口编排代码（通常不放业务实现）
src/core/     配置、Registry、日志和随机种子
src/data/     Dataset、Splitter、ExperimentPlanner、DataLoader
src/model/    普通 PyTorch 模型与训练组件
src/trainer/  Lightning 适配层和 Trainer 工厂
docs/         使用、配置、开发和 API 文档
```

模型和数据集不放在 `project/`。新增模型应放在 `src/model/` 并注册到 `MODEL_REGISTRY`；新增 Dataset 应放在 `src/data/` 并注册到 `DATASET_REGISTRY`。`project/main.py` 只负责读取配置、组装组件和启动实验。
