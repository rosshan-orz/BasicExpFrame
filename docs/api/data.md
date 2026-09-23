# Data API

## 样本和 Dataset

`SampleDict` 是 TypedDict 提示，包含可选 `inputs`、`targets`、`metadata` 字段。推荐的实际样本至少包含 inputs 与 targets。

`BaseDataset(file_path=None, transform=None)` 继承 PyTorch Dataset。传入路径时会验证路径存在；子类实现 `__len__`、`__getitem__`。`NpzDataset(file_path, transform=None, input_key="inputs", target_key="targets")` 是内置 NPZ 实现，支持 x/y 回退键名。

Registry：`DATASET_REGISTRY`、`SPLITTER_REGISTRY`，来自 `src.data`。

## Splitters

抽象 API：

```python
splitter(dataset) -> (train_dataset, valid_dataset, test_dataset)
splitter.split_indices(size, test_ratio=None) -> (train_indices, valid_indices, test_indices)
splitter.split_sequence(values, test_ratio=None) -> (train_values, valid_values, test_values)
```

`RandomSplitter(train_ratio=0.75, valid_ratio=0.125, test_ratio=0.125, seed=None)` 用 `random.Random(seed)` 隔离随机状态。`SequentialSplitter(...)` 按原顺序切分；两者都返回 PyTorch `Subset`。

## `Experiment`

冻结 dataclass：`Experiment(name, train, valid, test, metadata={})`。三个 split 是普通 Dataset；`subject_metadata` 返回 metadata，`train_subjects`、`valid_subjects`、`test_subjects` 返回对应 subject 元数据 tuple。

## `ExperimentPlanner`

构造参数：

```python
ExperimentPlanner(
    root=None, *, experiment_type="subject_dependent", splitter=None,
    dataset_factory=None, dataset_cls=None, dataset_config=None,
    subjects=None, file_pattern="*.npz", seed=None,
)
```

- `from_config(config)`：从根配置或 `data` 子配置构建 planner。
- `discover_subject_files() -> dict[str, Path]`：按 stem 发现文件并验证所选 subjects。
- `plan() -> list[Experiment]`、`iter(planner)`：生成实验对象。
- 类型为 `subject_dependent`、`cross_subject` 或 `leave_one_subject_out`。

`dataset_factory(path)` 可用于自定义数据读取；否则通过 `dataset_config.name/params` 从 Dataset Registry 构建，默认 `NpzDataset`。`dataset_cls` 当前保留作兼容扩展位，尚未使用。

## DataLoader

- `build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0, seed_worker=True, pin_memory=None, **kwargs)` 返回普通 PyTorch DataLoader。
- `build_loaders(train, valid, test, **kwargs)` 返回三份 loader，只有 train 默认 shuffle。
