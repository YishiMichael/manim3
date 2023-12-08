from __future__ import annotations


from enum import Enum
from typing import (
    Self,
    TypedDict,
    Unpack
)

from ..timelines.timeline.conditions import Condition
from .toplevel import Toplevel


class EventType(Enum):
    KEY_PRESS = 0
    KEY_RELEASE = 1
    MOUSE_MOTION = 2
    MOUSE_DRAG = 3
    MOUSE_PRESS = 4
    MOUSE_RELEASE = 5
    MOUSE_SCROLL = 6


class EventKwargs(TypedDict, total=False):
    symbol: int
    modifiers: int
    buttons: int
    x: int
    y: int
    dx: int
    dy: int
    scroll_x: float
    scroll_y: float


class Event:
    __slots__ = (
        "_event_type",
        "_event_kwargs"
    )

    def __init__(
        self: Self,
        event_type: EventType,
        **kwargs: Unpack[EventKwargs]
    ) -> None:
        super().__init__()
        self._event_type: EventType = event_type
        self._event_kwargs: EventKwargs = kwargs

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        target_kwargs = self._event_kwargs
        source_kwargs = event._event_kwargs
        return (
            self._event_type is event._event_type
            and all(
                (target_value := target_kwargs.get(field_name)) is None
                or (source_value := source_kwargs.get(field_name)) is None
                or target_value == (target_value & source_value if masked else source_value)
                for field_name, masked in (
                    ("symbol", False),
                    ("modifiers", True),
                    ("buttons", True),
                )
            )
        )

    def captured(
        self: Self
    ) -> EventCapturedCondition:
        return EventCapturedCondition(self)


class EventCapturedCondition(Condition):
    __slots__ = (
        "_event",
        "_captured_event"
    )

    def __init__(
        self: Self,
        event: Event
    ) -> None:
        super().__init__()
        self._event: Event = event
        self._captured_event: Event | None = None

    def judge(
        self: Self
    ) -> bool:
        captured_event = Toplevel._get_window().capture_event(self._event)
        if captured_event is not None:
            self._captured_event = captured_event
            return True
        return False

    def get_captured_event(
        self: Self
    ) -> Event | None:
        return self._captured_event
