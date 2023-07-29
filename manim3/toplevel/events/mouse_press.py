from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MousePress(Event):
    x: int
    y: int
    buttons: int
    modifiers: int
