# BasicExpFrame

基于 YAML、PyTorch 和 PyTorch Lightning 的实验框架。Dataset、Splitter 和模型保持普通 PyTorch 实现；Lightning 只负责训练执行。支持 subject-dependent、cross-subject 和 leave-one-subject-out（LOSO）实验。

## 快速开始

```bash
uv sync
uv run pytest tests/
uv run python project/main.py --config config/template.yaml
```

`config/template.yaml` 是不依赖实验数据的 smoke 配置。要训练模型，请参考[使用指南](docs/usage.md)创建数据配置。

## 文档

- [使用指南](docs/usage.md)：安装、数据格式、运行实验、CUDA、输出结构
- [配置参考](docs/configuration.md)：完整 YAML 字段说明与示例
- [开发与扩展](docs/development.md)：添加 Dataset、Splitter、模型与组件
- API： [Core](docs/api/core.md) · [Data](docs/api/data.md) · [Model](docs/api/model.md) · [Trainer](docs/api/trainer.md)
- [路线图与阶段记录](ROADMAP.md)
