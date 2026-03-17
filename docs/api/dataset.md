# Dataset API Reference

本指南详细说明了 BasicExpFrame 中数据流相关组件的 API 签名。所有的自定义数据集类应当基于此文档进行继承和实现。

---

## `SampleDict`

`SampleDict` 是一个用于规范类型提示 (Type Hint) 的字典协议，定义了单个样本及批次数据的期望返回格式。

```python
SampleDict = Dict[str, Union[Tensor, Dict[str, Tensor], Dict[str, Any]]]
```

* **说明**:
  * 框架推荐返回的字典至少包含 `inputs` 与 `targets` 两个主要键值。
  * `inputs`: 模型 `forward` 的输入数据。可以是一个 `Tensor`，也可以是嵌套的 `Dict[str, Tensor]`（供多模态输入解包）。
  * `targets`: 用于计算损失的真实标签（Ground Truth）。

---

## `BaseDataset`

```python
class src.dataset.base_dataset.BaseDataset(file_path: Union[str, Path], transform: Optional[Callable] = None)
```

作为所有自定义数据集的抽象基类。该类继承自 `torch.utils.data.Dataset`，并为框架配置驱动的数据读取提供了标准初始化接口。

在此类及其子类之上，必须使用 `@DATASET_REGISTRY.register()` 装饰器。

### Parameters

* **file_path** (`str` or `pathlib.Path`) – 单个受试者或切分后的目标数据的物理路径（通常对应配置字典中的 `data.dataset.params.file_path`）。基类的 `__init__` 会自动校验该路径是否存在。
* **transform** (`Callable`, optional) – 应用于单个样本的可选变换（如数据增强、重新采样）。默认值为 `None`。

### Methods

#### `__len__() -> int`
*(Abstract Method)* 返回数据集中样本的总数。

* **Returns:** 样本个数。
* **Return type:** `int`

#### `__getitem__(index: int) -> SampleDict`
*(Abstract Method)* 根据指定的 `index` 提取单个数据样本，并封装为 `SampleDict` 返回。

* **Parameters:**
  * **index** (`int`) – 目标样本的索引。
* **Returns:** 包含输入数据和标签字典的结构化数据，供 DataLoader 进一步打包为批次。
* **Return type:** `SampleDict`

---

## `BaseSplitter`

```python
class src.dataset.splitters.base.BaseSplitter()
```

用于定义数据集拆分策略的抽象基类。它将一个完整的数据集（或 `ConcatDataset`）切分为训练集、验证集和测试集。

在此类及其子类之上，必须使用 `@SPLITTER_REGISTRY.register()` 装饰器。

### Methods

#### `__call__(dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]`
*(Abstract Method)* 执行切分逻辑。框架在实例化该类后，会通过 `__call__` 函数式调用对象。

* **Parameters:**
  * **dataset** (`torch.utils.data.Dataset`) – 等待划分的完整父数据集（在跨受试者实验中，这可能是一个包含所有被试数据的巨型组合集）。
* **Returns:** 一个必须包含三个元素的元组，依次代表 `(train_dataset, valid_dataset, test_dataset)`。例如，当不需要测试集时，第三个元素可以为空的切片或列表。
* **Return type:** `Tuple[Dataset, Dataset, Dataset]`

---

## `DataBuilder`

```python
class src.dataset.builder.DataBuilder()
```

用于调度以上各个元素，并负责执行具体的加载策略（由 `yaml` 文件中的 `experiment_type` 控制）。它是 `main.py` 内部直接调用的数据集构建和管理大管家。

### Methods

#### `build_experiments(config: Box) -> Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]`
*(Static Method)* 生成各种受试者组合实验环境的核心入口。

* **Parameters:**
  * **config** (`Box`) – 包含 `data` 配置命名空间的 `yaml` 根字典。
* **Returns:** 包含完整组装信息的生成器对象。每一次迭代，即表示一次完全独立的验证周期。返回元组中第一个元素为可读的字符串（如实验重现名 `subject_dependent_fold_0`），后三者为已经过 `num_workers` 多进程和对应 `batch_size` 打包就绪的标准化 PyTorch DataLoader 容器。
* **Return type:** `Iterator[Tuple[str, DataLoader, DataLoader, DataLoader]]`
* **说明**：通过解析 `experiment_type` 进行内部逻辑分发，原生支持了诸如 `subject_dependent` (单一受试者相关), `cross_subject` (跨受试者大拼盘) 以及 `leave_one_subject_out` (留一法) 实验机制。

> **策略扩展指南**: 如果需要在 `DataBuilder` 内增设私有的实验交叉重组循环（比如指定留K法等），只需为其补充如 `_run_leave_k_out` 的专职方法，并映射进 `build_experiments` 下的 `strategies` 分发字典即可。

---

## `RandomSplitter`

```python
class src.dataset.splitters.random_splitter.RandomSplitter(train_ratio: float = 0.75, valid_ratio: float = 0.125, test_ratio: float = 0.125, seed: Optional[int] = None)
```
继承自 `BaseSplitter` 的实体划分器类，已默认注册为 `"RandomSplitter"`。

### Parameters

* **train_ratio** (`float`, optional) – 训练集所占总体样本的比例，默认为 `0.75`。
* **valid_ratio** (`float`, optional) – 验证集所占总体样本的比例，默认为 `0.125`。
* **test_ratio** (`float`, optional) – 测试集所占总体样本的比例，默认为 `0.125`。
* **seed** (`int`, optional) – 用于保证每次随机采样的切分结果完全可复现的全局随机种子。

### Methods

#### `__call__(dataset: Dataset) -> Tuple[Dataset, Dataset, Dataset]`
打乱全部数据的可用索引 `indices`，并按给定比例通过 `torch.utils.data.Subset` 将数据集物理切分为三份。

* **Parameters:**
  * **dataset** (`torch.utils.data.Dataset`) – 被处理的父数据集。
* **Returns:** 封装好的子数据集元组 `(train_dataset, valid_dataset, test_dataset)`。
* **Return type:** `Tuple[Dataset, Dataset, Dataset]`
* **Raises:** `ValueError` - 当初始化时传入的三个 ratio 之和不等于 `1.0` 时触发。
