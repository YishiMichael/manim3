from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class KeyRelease(Event):
    symbol: int
    modifiers: int
