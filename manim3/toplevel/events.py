from dataclasses import dataclass


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class Event:
    pass


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class KeyPress(Event):
    symbol: int
    modifiers: int


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class KeyRelease(Event):
    symbol: int
    modifiers: int


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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MouseDrag(Event):
    x: int
    y: int
    dx: int
    dy: int
    buttons: int
    modifiers: int


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


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class MouseRelease(Event):
    x: int
    y: int
    buttons: int
    modifiers: int


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
