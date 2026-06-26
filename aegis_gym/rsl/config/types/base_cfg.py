from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints, get_args, get_origin


T = TypeVar("T")


class BaseCfg(ABC):
    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any] | None = None) -> T:
        """
        Recursively build dataclass from partial dict.
        Missing values use dataclass defaults.
        """
        data = data or {}
        hints = get_type_hints(cls)
        kwargs = {}
        for f in fields(cls):
            if f.name not in data:
                continue
            kwargs[f.name] = BaseCfg._deserialize(data[f.name], hints[f.name])
        return cls(**kwargs)

    @classmethod
    def _deserialize(cls, value: Any, typ: Any) -> Any:
        origin = get_origin(typ)

        # BaseCfg derivatives
        if is_dataclass(typ):
            return typ.from_dict(value)

        # pathlib.Path
        if typ is Path:
            return Path(value)

        # list[T]: YAML may give {0: ..., 1: ..., 2: ...} instead of a real list
        if origin is list:
            (elem_type,) = get_args(typ)
            if isinstance(value, dict):  # numeric-keyed dict -> sorted list
                value = [v for _, v in sorted(value.items())]

            return [BaseCfg._deserialize(v, elem_type) for v in value]

        # dict[K, V]
        if origin is dict:
            key_type, val_type = get_args(typ)
            return {
                key_type(k): BaseCfg._deserialize(v, val_type) for k, v in value.items()
            }

        # Any other (base) type
        return value

    def as_dict(self) -> dict:
        """
        Recursively convert dataclass (and nested dataclasses) to dict.
        """
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} is not a dataclass")

        result = {}
        for f in fields(self):
            value = getattr(self, f.name)
            result[f.name] = self._serialize_value(value)
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Handle nested dataclasses, lists, tuples, and dicts."""
        if is_dataclass(value) and not isinstance(value, type):
            return value.as_dict()
        elif isinstance(value, dict):
            return {k: BaseCfg._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            serialized = [BaseCfg._serialize_value(v) for v in value]
            return type(value)(serialized)
        elif isinstance(value, Path):
            return str(value)
        else:
            return value


@dataclass(slots=True, frozen=True)
class ToggleCfg(BaseCfg):
    enabled: bool = False
