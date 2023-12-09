from __future__ import annotations


from typing import (
    Literal,
    Required,
    Self,
    TypedDict,
    Unpack
)

from .toplevel import Toplevel


type EventType = Literal[
    "key_press",
    "key_release",
    "mouse_motion",
    "mouse_drag",
    "mouse_press",
    "mouse_release",
    "mouse_scroll"
]


class Event(TypedDict, total=False):
    event_type: Required[EventType]
    symbol: int
    modifiers: int
    buttons: int
    x: int
    y: int
    dx: int
    dy: int
    scroll_x: float
    scroll_y: float


class EventCapturer:
    __slots__ = (
        "_event_kwargs",
        "_captured_event"
    )

    def __init__(
        self: Self,
        **kwargs: Unpack[Event]
    ) -> None:
        super().__init__()
        self._event_kwargs: Event = kwargs
        self._captured_event: Event | None = None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        target = self._event_kwargs
        if (
            target["event_type"] == event["event_type"]
            and all(
                (target_value := target.get(field_name)) is None
                or (source_value := event.get(field_name)) is None
                or target_value == (target_value & source_value if masked else source_value)
                for field_name, masked in (
                    ("symbol", False),
                    ("modifiers", True),
                    ("buttons", True),
                )
            )
        ):
            self._captured_event = event
            return True
        return False

    def captured(
        self: Self
    ) -> bool:
        return self._captured_event is not None or Toplevel._get_window().capture_event(self)

    def get_captured_event(
        self: Self
    ) -> Event | None:
        return self._captured_event
