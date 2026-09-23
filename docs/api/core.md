# Core API

## `Registry`

位置：`src.core.registry.Registry(name: str)`。

- `register(name=None)`：返回类/函数装饰器；未提供名称时使用对象名，重复名称抛 `KeyError`。
- `get(name)`：取出对象；不存在时抛 `KeyError` 并列出可用项。
- `build(config, **kwargs)`：`config` 为名称字符串或 `{name, params}` 字典；额外 kwargs 覆盖 `params` 同名值，并调用对象构造函数。
- 支持 `name in registry`、`len(registry)`、`keys()`、`items()`。

## 配置加载

`load_config(path: str | Path) -> box.Box` 与 `ConfigParser.load(path)` 等价。返回嵌套属性式访问的冻结 Box。文件不存在抛 `FileNotFoundError`；空文件或根节点不是 mapping 抛 `ValueError`。

## 日志与种子

- `BaseLogger(log_dir, name=None, level=logging.INFO)`：创建目录，同时写控制台与 `train.log`；提供 `debug/info/warning/error` 和 `close()`。
- `set_seed(seed: int, deterministic=False) -> None`：设置 Python、NumPy、PyTorch 和 CUDA 随机种子；deterministic 为真时开启 cuDNN deterministic 并关闭 benchmark。
- `worker_init_fn(worker_id: int) -> None`：按 PyTorch worker seed 初始化 Python 与 NumPy RNG，DataLoader builder 默认在 worker 数大于 0 时使用。
