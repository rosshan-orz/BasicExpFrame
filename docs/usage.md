# 使用指南

## 环境

项目以 `pyproject.toml` 和 `uv.lock` 管理依赖，要求 Python 3.10 或更新版本。

```bash
uv sync
uv run pytest tests/
uv run python project/main.py --config config/template.yaml
```

如果只想运行核心 smoke 流程，模板命令不需要数据文件。

## NPZ 数据格式

每个 `.npz` 文件代表一个 subject，放在 `data.root` 指定目录。内置 `NpzDataset` 默认读取 `inputs` 和 `targets` 两个数组，也接受 `x` 和 `y` 作为兼容键名。两个数组的第 0 维必须相同。

```python
import numpy as np

np.savez(
    "data/sub01.npz",
    inputs=np.random.randn(100, 16).astype("float32"),
    targets=np.random.randint(0, 2, size=100).astype("int64"),
)
```

`inputs` 与 `targets` 会按索引组成样本。浮点型输入会转成 `float32`，整数标签保持整数类型。模型输入通常形状为 `[batch, features...]`；分类标签通常为 `[batch]`。

## 运行训练

复制或改写完整示例配置（参见[配置参考](configuration.md)），设置 `data.root`、模型参数和训练轮数，再运行：

```bash
uv run python project/main.py --config config/my_experiment.yaml
```

入口按配置完成以下工作：设置全局种子、规划一个或多个 Experiment、构建模型及训练组件、调用 Lightning `fit`，然后使用最佳 checkpoint 测试。`config/template.yaml` 未配置 `data`，所以只执行 core smoke，不会开始训练。

## 设备与 CUDA

CPU 配置：

```yaml
device: cpu
```

CUDA 配置建议显式指定 Lightning accelerator：

```yaml
device: cuda
trainer:
  accelerator: cuda
  devices: 1
```

CUDA 主机还必须使用能识别 CUDA 的 PyTorch 构建和兼容的 NVIDIA 驱动。可先检查：

```bash
uv run python -c 'import torch; print(torch.cuda.is_available(), torch.version.cuda)'
```

代码不会把 CUDA Toolkit/驱动安装到系统，也不会在 CUDA 不可用时静默退回 CPU；明确写 `accelerator: cuda` 时，Lightning 会在设备不可用时报告错误。

## 输出

运行产物位于 `output_dir / experiment_name / timestamp / experiment.name`：

- `train.log`：框架 Python 日志
- `checkpoints/`：Lightning checkpoint
- `lightning/`：Lightning CSV 日志

每个 Experiment 使用独立目录。当前入口使用 CSVLogger，并不默认生成 TensorBoard event 文件。
