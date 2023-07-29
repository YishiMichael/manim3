from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class KeyPress(Event):
    symbol: int
    modifiers: int
