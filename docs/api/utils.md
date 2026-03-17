# Utils API Reference

本指南详细说明了 BasicExpFrame 中通用工具集（Utils）的 API 签名。包括了日志追踪、注册表管理、检查点存取等核心外围组件。

---

## `Registry`

```python
class src.utils.registry.Registry(name: str)
```

实现名称到对象的映射字典，用于支持配置驱动（Configuration-driven）和模块解耦的动态实例化设计。

### Parameters

* **name** (`str`) – 注册表的全局标识名（例如 `"models"`, `"dataset"`）。

### Methods

#### `register(name: Optional[str] = None) -> Callable[[Type[Any]], Type[Any]]`
装饰器方法。将一个目标类（或函数）注册进当前的 Registry 实例池中。

* **Parameters:**
  * **name** (`str`, optional) – 注册时的别名。若不提供，将默认使用被装饰对象的 `__name__` 属性。
* **Returns:** 返回一个装饰器函数，该函数接收类并原样返回以保证无侵入注册。
* **Return type:** `Callable`
* **Raises:** `KeyError` – 如果该名字已经被注册过，则报错。

#### `get(name: str) -> Type[Any]`
根据注册名字符串，提取出对应的类或函数句柄（未实例化）。

* **Parameters:**
  * **name** (`str`) – 想要提取的模块名称。
* **Returns:** 对应的 Python 类或函数本身。
* **Return type:** `Type[Any]`
* **Raises:** `KeyError` – 如果查询的名字在这个注册表中不存在。

#### `build(config: Any, **kwargs: Any) -> Any`
这是 `Registry` 在运行时的核心动作接口。它接收包含 `name` 和 `params` 的结构化字典，利用反射动态实例化对象。

* **Parameters:**
  * **config** (`Any`) – 可以是字符串（仅依靠默认参数初始化），或是包含键 `"name"` 和（可选的）键 `"params"`（或解析后的 `Box` 类型等价物）的字典。
  * **kwargs** (`Any`) – 运行时的动态增补参数，这些增补参数在合并时具有最高优先级，可以覆盖 `config.params` 中的同名静态参数。
* **Returns:** 根据配置组装好的已实例化对象。
* **Return type:** `Any`

---

## `BaseLogger`

```python
class src.utils.logger.BaseLogger(log_dir: Union[str, Path], config: Box)
```

封装了 Python 原生 `logging` 模块与 PyTorch 的 `TensorBoard SummaryWriter`，为框架提供了控制台信息打印、物理文件落盘以及多维曲线追踪的三合一日志总控。

### Parameters

* **log_dir** (`str` or `pathlib.Path`) – 物理文件持久化目录的绝对/相对路径。在其下通常会默认生成 `train.log` 及 `tb_logs/` 以分别存放明文日志与事件。
* **config** (`Box`) – 从 YAML 解析出的只读环境配置字典（预备项，当前实现中作为依赖挂载）。

### Methods

#### `info(msg: str) -> None`
将一条等级为 `INFO` 的纯文本信息推送给绑定的所有 Handler（例如：同时在终端控制台和 `train.log` 中追加记录）。

#### `warning(msg: str, *args: Any, **kwargs: Any) -> None`
推送警告级别的日志记录。

#### `error(msg: str, *args: Any, **kwargs: Any) -> None`
推送错误级别的日志记录。

#### `log_metrics(metrics: Dict[str, float], step: int) -> None`
批量写入带有步骤标识的标量指标（Scalars）。

* **Parameters:**
  * **metrics** (`Dict[str, float]`) – 各类被评估出的最新指标字典（例如：`{'train/loss': 0.1, 'val/acc': 0.95}`）。方法内部会自动剔除非纯标量值以免崩溃。
  * **step** (`int`) – 绑定记录所处的全局 Step 或 Epoch 横坐标值。
* **说明**: 会同时以字典形式调用 `info()` 保存在明文日志，并通过 `writer.add_scalar()` 更新给 TensorBoard。

#### `close() -> None`
清理生命周期环境：主动释放并安全关闭 TensorBoard 的事件写句柄；随后关闭并移除基础 `logging` 模块上派生的所有 FileHandlers 防治文件系统锁死溢出。

---

## `Checkpoint Utilities`

包含在文件 `src/utils/checkpoint.py` 下的系列函数。主要提供模型状态与优化器持久化的快速封装。

#### `save_checkpoint(state: Dict[str, Any], save_dir: Union[str, Path], file_name: str = "last.pth", is_best: bool = False) -> None`

* **Parameters:**
  * **state** (`Dict[str, Any]`) – 当前周期需要被序列化的运行快照字典（往往由 Trainer 包装产生，内含 `epoch`, `model_state_dict`, `optimizer_state_dict` 等等）。
  * **save_dir** (`str` or `Path`) – 提供存放检查点的物理外层目录路径。
  * **file_name** (`str`, optional) – 落盘的文件名称，默认覆写为 `"last.pth"`。
  * **is_best** (`bool`, optional) – 标志位。如果为真，将会额外把刚才落盘的文件再并列拷贝/转出存盘一份为 `save_dir/"best.pth"` 供历史高光回溯。

#### `load_checkpoint(file_path: Union[str, Path], map_location: Optional[Union[str, torch.device]] = None) -> Dict[str, Any]`

* **Parameters:**
  * **file_path** (`str` or `Path`) – 目标权重存档的确切全路径。
  * **map_location** (`str` or `torch.device`, optional) – 指示张层挂载去向（常用来解决把用 GPU 计算的 checkpoint 重载进 CPU 以供本地推理所造成的不匹配异常）。
* **Returns:** 可用于装载 `load_state_dict()` 的结构化原装环境字典。
* **Return type:** `Dict[str, Any]`
* **Raises:** `FileNotFoundError` - 如果指定断点文件不存在。

---

## `ConfigParser`

```python
class src.utils.config_parser.ConfigParser()
```

用于处理配置驱动的核心解析类。负责将 YAML 文件映射为支持跨层级点号访问的冻结字典对象。

### Methods

#### `load(path: Union[str, Path]) -> Box`
*(Static Method)* 从指定路径读取 YAML 并转化为强类型的 `Box` 对象。

* **Parameters:**
  * **path** (`str` or `Path`) – YAML 配置文件的所在位置。
* **Returns:** 冻结的（禁止运行时覆写） `Box` 实例。这使得我们在代码中可以安全地通过形如 `config.trainer.epochs` 的形式调用参数。
* **Return type:** `box.Box`
* **Raises:** `FileNotFoundError` - 如果给定的配置路径无法被解析。
