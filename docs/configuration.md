# 配置参考

配置文件为 YAML。入口通过 `load_config` 加载冻结的 `Box`；业务代码应读取配置，不应在运行中修改它。

## 可运行训练示例

该配置假设 `data/sub01.npz` 已按[使用指南](usage.md)准备好。保存为 `config/example.yaml` 后可用 `uv run python project/main.py --config config/example.yaml` 启动。

```yaml
experiment_name: demo
seed: 42
device: cpu
output_dir: ./output

data:
  root: ./data
  file_pattern: "*.npz"
  experiment_type: subject_dependent
  subjects: [all]
  dataset:
    name: NpzDataset
    params:
      input_key: inputs
      target_key: targets
  splitter:
    name: RandomSplitter
    params:
      train_ratio: 0.7
      valid_ratio: 0.15
      test_ratio: 0.15
      seed: 42
  loader:
    batch_size: 32
    num_workers: 0
    pin_memory: false

model:
  name: MLP
  params:
    input_dim: 16
    hidden_dim: 64
    num_classes: 2
    dropout: 0.1
loss:
  name: CrossEntropyLoss
  params: {}
optimizer:
  name: AdamW
  params:
    lr: 0.001
    weight_decay: 0.0001
scheduler:
  name: StepLR
  params:
    step_size: 10
    gamma: 0.5
metrics:
  - name: Accuracy
    params:
      num_classes: 2
trainer:
  max_epochs: 20
  accelerator: auto
  devices: 1
  precision: 32-true
  monitor: val/loss
  monitor_mode: min
  callbacks:
    early_stopping:
      monitor: val/loss
      mode: min
      patience: 5
```

## 顶层字段

| 字段 | 类型 | 默认/说明 |
|---|---|---|
| `experiment_name` | string | 实验输出目录名；必需 |
| `seed` | integer | 全局 Python/NumPy/PyTorch 随机种子 |
| `device` | string | `cpu` 时入口默认设置 Lightning `accelerator=cpu, devices=1`；其他值不直接映射。CUDA 请在 `trainer` 中显式设置 `accelerator: cuda` |
| `output_dir` | path | 输出根目录；相对路径相对于仓库根目录 |
| `data` | mapping | 数据与实验规划；缺失时只运行 smoke |
| `model` | mapping | 模型 Registry 配置，训练时必需 |
| `loss` | mapping | Criterion Registry 配置；入口默认 CrossEntropyLoss |
| `optimizer` | mapping | Optimizer Registry 配置；入口默认 Adam(lr=0.001) |
| `scheduler` | mapping/null | Scheduler 配置，可省略或设为 null |
| `metrics` | list | Metric 配置列表，可省略 |
| `trainer` | mapping | Lightning Trainer 选项，也接受 `epochs`、`grad_clip`、`use_amp` 别名 |

组件配置的一般形式为 `{name: Registry名称, params: 构造参数}`。`Registry.build` 也接受仅有名称的字符串（即无构造参数）。

## `data`

| 字段 | 说明 |
|---|---|
| `root` | subject 文件目录，必需 |
| `file_pattern` | 文件 glob，默认 `*.npz` |
| `experiment_type` | `subject_dependent`、`cross_subject` 或 `leave_one_subject_out` |
| `subjects` | subject stem 列表；`[all]` 或省略表示选中所有发现的文件。兼容旧字段 `test_subjects` |
| `dataset` | Dataset Registry 配置；默认内置 `NpzDataset` |
| `splitter` | Splitter 配置；默认 `RandomSplitter` |
| `loader.batch_size` | DataLoader batch size，默认 1 |
| `loader.num_workers` | worker 数，默认 0 |
| `loader.pin_memory` | 是否启用 pinned memory；省略时按 CUDA 是否可用决定 |

cross-subject 按 subject 切出 train/valid/test；LOSO 的留出 subject 只作为 test，其余 subject 再分为 train/valid。至少需要两个 subject。自定义 Dataset 时可以在 Python API 中传入 `dataset_factory`。

## Splitter 参数

`RandomSplitter` 和 `SequentialSplitter` 接受 `train_ratio`、`valid_ratio`、`test_ratio`、`seed`，比例须在 `[0,1]` 且总和为 1。随机切分使用局部随机源，不修改全局 `random` 状态。顺序切分保持输入顺序。

`SequentialSplitter` 额外接受旧版兼容参数 `ratio`：它表示 train 比例，剩余部分平均分配给 valid 和 test；新配置建议使用三个明确比例。

## `trainer`

传给 `lightning.pytorch.Trainer` 的有效参数可直接写入 `trainer`，例如 `max_epochs`、`accelerator`、`devices`、`precision`、`fast_dev_run`、`gradient_clip_val`。框架别名：

| 框架字段 | Lightning 字段/行为 |
|---|---|
| `epochs` | `max_epochs` |
| `grad_clip` | `gradient_clip_val` |
| `use_amp: true` | 若未指定 `precision`，设置为 `16-mixed` |

入口会读取 `monitor`/`monitor_mode` 配置来设置默认 ModelCheckpoint 监控项，默认 `val/loss` / `min`。`callbacks` 可设置 `checkpoint`、`early_stopping`、`learning_rate_monitor`，具体构造参数见 [Trainer API](api/trainer.md)。
