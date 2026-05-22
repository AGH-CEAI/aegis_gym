from abc import ABC
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Type, TypeVar, get_type_hints

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
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
            else:
                kwargs[f.name] = value

        return cls(**kwargs)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ToggleCfg(BaseCfg):
    enabled: bool = False
