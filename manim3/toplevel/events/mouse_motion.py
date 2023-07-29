from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MouseMotion(Event):
    x: int
    y: int
    dx: int
    dy: int
