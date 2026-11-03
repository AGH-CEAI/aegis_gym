from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """
    Describes the normalized [-1, 1] actions of an environment for their preview: `names`,
    the `units` of the physical commands and the `scales` mapping the actions to them.
    `joystick` holds the indices of an action pair shown in the joystick (XY) panel.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    scales: tuple[float, ...]
    joystick: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not len(self.names) == len(self.units) == len(self.scales):
            raise ValueError(
                f"The action spec has {len(self.names)} names, {len(self.units)} units "
                f"and {len(self.scales)} scales."
            )
        if self.joystick is not None and not all(
            0 <= i < len(self.names) for i in self.joystick
        ):
            raise ValueError(
                f"The joystick actions {self.joystick} are out of the {len(self.names)} actions."
            )

    @classmethod
    def default(cls, num_actions: int) -> "ActionSpec":
        """Unnamed, unitless actions `a0..aN`."""
        return cls(
            names=tuple(f"a{i}" for i in range(num_actions)),
            units=("",) * num_actions,
            scales=(1.0,) * num_actions,
        )

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ActionSpec":
        joystick = data.get("joystick")
        return cls(
            names=tuple(data["names"]),
            units=tuple(data["units"]),
            scales=tuple(data["scales"]),
            joystick=tuple(joystick) if joystick is not None else None,
        )
