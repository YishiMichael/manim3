from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import Self

import attrs

from ..timelines.timeline.condition import Condition
from .toplevel import Toplevel


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


@attrs.frozen(kw_only=True)
class Event(ABC):
    @abstractmethod
    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        pass

    @classmethod
    def _match(
        cls: type[Self],
        required_value: int | None,
        value: int | None,
        *,
        masked: bool
    ) -> bool:
        return (
            required_value is None
            or value is None
            or required_value == (value & required_value if masked else value)
        )

    def captured(
        self: Self
    ) -> EventCapturedCondition:
        return EventCapturedCondition(self)
