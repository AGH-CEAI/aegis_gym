from collections.abc import Callable

from .base_env import BaseEnv

_ENV_REGISTRY: dict[str, type[BaseEnv]] = {}


def register_env(name: str) -> Callable[[type[BaseEnv]], type[BaseEnv]]:
    def decorator(cls: type[BaseEnv]) -> type[BaseEnv]:
        _ENV_REGISTRY[name] = cls
        return cls

    return decorator


def get_env_class(name: str) -> type[BaseEnv]:
    if name not in _ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{name}'. Available: {available_envs()}")
    return _ENV_REGISTRY[name]


def available_envs() -> list[str]:
    return list(_ENV_REGISTRY)
