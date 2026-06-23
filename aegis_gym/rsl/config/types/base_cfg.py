from abc import ABC
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar, get_type_hints


T = TypeVar("T")


class BaseCfg(ABC):
    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any] | None = None) -> T:
        """
        Recursively build dataclass from partial dict.
        Missing values use dataclass defaults.
        """
        data = data or {}

        kwargs = {}
        hints = get_type_hints(cls)

        for f in fields(cls):
            field_type = hints[f.name]

            if f.name not in data:
                continue

            value = data[f.name]

            if is_dataclass(field_type):
                kwargs[f.name] = field_type.from_dict(value)
            elif isinstance(field_type, Path):
                kwargs[f.name] = Path(value)
            else:
                kwargs[f.name] = value

        return cls(**kwargs)

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
