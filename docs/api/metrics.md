# Metrics API Reference

本指南详细说明了 BasicExpFrame 中评价指标的 API 签名。所有的自定义指标（如 Accuracy、F1-Score、MSE 等）应当遵守以下接口约束，以保证它们能被 `MetricManager` 自动采集和格式化。

---

## `BaseMetric`

```python
class src.metrics.abstract.BaseMetric(name: str)
```

作为所有评价指标定义的抽象基类。该类规定了指标在整个 Epoch 运转生命周期内重置、累加更新、最终求解的三大必备行为。

在此类及其子类之上，必须使用 `@METRIC_REGISTRY.register()` 装饰器。

### Parameters

* **name** (`str`) – 给当前评价指标赋予的全局唯一标识名称（例如 `"acc"`, `"f1"`, `"mse"`）。该名称将作为最终写入 TensorBoard 和控制台日志的根键值。

### Methods

#### `reset() -> None`
*(Abstract Method)* 重置指标的内部中间状态变量。

* **说明**: 该方法由 `MetricManager` 在**每个 Epoch 的起始时刻**（不论是 Training 还是 Validation 阶段）自动调用。开发者需在此方法中将累加器清零（例如 `self.correct = 0`, `self.total = 0`）。

#### `update(outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> None`
*(Abstract Method)* 使用当前批次（Batch）的模型预测值与真实标签更新内部累计状态。

* **说明**: 由 `MetricManager` 在**每个 Batch 前向传播完毕**后自动调用。需要注意，由于该函数在训练的极内环被高频调用，内部最好避免分配过大的 Python 临时对象以防内存激增，且*不可在此处直接计算均值*，均值只应利用总数在最后除得。
* **Parameters:**
  * **outputs** (`Dict[str, Tensor]`) – 模型在当前 Batch 返回的结果字典（建议从中读取 `outputs.get('logits')`）。
  * **batch** (`Dict[str, Any]`) – 载有标注标签 `targets` 等元信息的原始 Batch 字典。

#### `compute() -> Dict[str, float]`
*(Abstract Method)* 根据从 Epoch 开始累计至今的内部状态变量，计算最终的评估分值。

* **说明**: 在**单个 Epoch 迭代所有 Batch 完全结束**后，由 `MetricManager` 统一调用，一次性产出此轮最终打分。
* **Returns:** 必须返回一个以字符串为键，基础浮点为值的字典格式。此字典会平铺并挂载到最终日志板上。如果不按规范返回此类型，将引发异常。
* **Return type:** `Dict[str, float]`

---

## `MetricManager`

```python
class src.metrics.manager.MetricManager(metric_config: List[Union[str, Dict[str, Any]]])
```

指标的全局代理人。该类统一保管 `yaml` 配置档中所请求实例化的所有评价指标对象。利用该中心化代理类，框架得以做到只需一个入口函数调用，即能驱动旗下管理的所有 `BaseMetric` 并行运转。

### Parameters

* **metric_config** (`List[Union[str, Dict[str, Any]]]`) – 包含各个 metric 初始化详情的配置列表（通常映射的是 yaml 中定义的注册列表）。在初始化阶段，若发现同名指标冲突将会提前报 `ValueError`。

### Methods

#### `reset() -> None`
透明转发接口，它遍历内部持有的所有已被实例化的 Metric，并级联触发其下属对象的 `.reset()`。

#### `update(outputs: Dict[str, Tensor], batch: Dict[str, Any]) -> None`
数据分发中转塔。收集每个 Batch 结算后的预测值与真实标签数据 `batch`，并原封不动地将其拷贝分发给手下所有的 `BaseMetric` 成员以同步各自的累计值。

#### `compute() -> Dict[str, float]`
最终汇总并平铺所有的统计运算结果。
* **Returns:** 将手下所有成员通过 `.compute()` 返回的心血（比如某个成员返回了 `{"acc": 0.95}`，另一个返回了 `{"mse": 0.05}`）解包、去重判断并合并为一个平铺大字典进行对外交付。
* **Return type:** `Dict[str, float]`
* **Raises:** `ValueError` - 当发现手下的两个指标返回了同名的键名（这会导致 TensorBoard 上的曲线完全覆写且错乱失真）时将立即截断程序。

