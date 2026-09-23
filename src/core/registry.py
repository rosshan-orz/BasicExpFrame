"""Generic registry for configuration-driven object construction."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar("T", bound=type)


class Registry:
    """A name-to-object registry supporting decorator registration and build."""

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    def register(self, name: Optional[str] = None) -> Callable[[T], T]:
        """Register a class or function under an optional alias."""

        def decorator(obj: T) -> T:
            key = name if name is not None else getattr(obj, "__name__", repr(obj))
            if key in self._registry:
                raise KeyError(
                    f"{key} is already registered in {self._name}. "
                    f"Available items: {sorted(self._registry.keys())}"
                )
            self._registry[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        """Get the registered object by name."""
        if name not in self._registry:
            raise KeyError(
                f"'{name}' is not registered in {self._name}. "
                f"Available items: {sorted(self._registry.keys())}"
            )
        return self._registry[name]

    def build(self, config: Any, **kwargs: Any) -> Any:
        """Build an object from a string or a dict-like config."""
        if isinstance(config, str):
            name = config
            params: Dict[str, Any] = {}
        elif isinstance(config, dict):
            name = config.get("name")
            if name is None:
                raise ValueError(
                    f"Config for {self._name} must contain a 'name' key. Got: {config}"
                )
            params = config.get("params", {})
            if params is None:
                params = {}
        else:
            raise TypeError(
                f"Unsupported config type for {self._name}: {type(config)}. "
                "Expected str or dict."
            )

        obj = self.get(name)
        merged_params = {**params, **kwargs}
        return obj(**merged_params)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __len__(self) -> int:
        return len(self._registry)

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()

    def __repr__(self) -> str:
        return (
            f"<Registry(name={self._name!r}, items={sorted(self._registry.keys())})>"
        )
