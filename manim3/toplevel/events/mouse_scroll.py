from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MouseScroll(Event):
    x: int
    y: int
    scroll_x: float
    scroll_y: float
