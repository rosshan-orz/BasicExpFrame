from typing import Callable, Type, Any, Dict, Optional

class Registry:
    """
    The registry that provides name -> object mapping, to support flexible instantiation based on configuration
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}

    def __len__(self) -> len:
        return len(self._module_dict)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}(name='{self._name}', "
            f"items={list(self._module_dict.keys())})>"
        )

    def register(self, name: Optional[str] = None) -> Callable[[Type[Any]], Type[Any]]:
        """
        A decorator to register a module. The module could be a function or a class.
        :param name: Name of the module.
        :return: a decorator to register the module
        """
        def decorator(func: Type[Any]) -> Type[Any]:
            key = name if name is not None else func.__name__
            if key in self._module_dict:
                raise KeyError(f"Module {key} already registered in {self._name}")
            self._module_dict[key] = func
            return func
        return decorator
    def get(self, name: str) -> Type[Any]:
        """
        Retrieve a registered class or function by its name.
        Raises KeyError if the name is not found.
        :param name: Name of the module
        :return: the actual Python class or function that was previously registered under the given name.
        """
        if name not in self._module_dict:
            raise KeyError(
                f"Module {name} not registered in {self._name}"
                f"Available items: {list(self._module_dict.keys())}"
            )
        return self._module_dict[name]
    def build(self, config: Any, **kwargs: Any) -> Any:
        """
        Look up the module by its name in the provided configuration file,
        and initialize it with params found in config.
        Build a module from its config and kwargs.
        :param config: config of the module
        :param kwargs: args of the module
        :return: An instantiated object of the registered class,
        or the result of calling the registered function.
        """
        if isinstance(config, str):
            name = config
            params = {}
        elif isinstance(config, dict):
            name = config.get("name")
            if name is None:
                raise ValueError("Config must contain a 'name' key for building.")
            params = config.get("params", {})
        else:
            raise TypeError(f"Unsupported config type: {type(config)}. Expected str or dict.")

        cls = self.get(name)
        # Combine params from config with any direct kwargs, with kwargs taking precedence
        final_params = {**params, **kwargs}
        return cls(**final_params)

# =============================================================================
# Centralized Registry Instances
# =============================================================================
# To avoid circular imports, all registry instances are defined here
# and should be imported from src.utils.registry by other modules.

MODEL_REGISTRY = Registry("models")
DATASET_REGISTRY = Registry("dataset")
METRIC_REGISTRY = Registry("metrics")
CRITERION_REGISTRY = Registry("criterion")
OPTIMIZER_REGISTRY = Registry("optimizers")
SCHEDULER_REGISTRY = Registry("schedulers")
SPLITTER_REGISTRY = Registry("splitters")