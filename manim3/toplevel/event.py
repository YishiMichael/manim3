from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass
from typing import Self

from ..timelines.timeline.condition import Condition
from .toplevel import Toplevel


class EventCapturedCondition(Condition):
    __slots__ = ("_event",)

    def __init__(
        self: Self,
        event: Event
    ) -> None:
        super().__init__()
        self._event: Event = event

    def judge(
        self: Self
    ) -> bool:
        return Toplevel.window.capture_event(self._event)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class Event(ABC):
    @abstractmethod
    def _capture(
        self: Self,
        event: "Event"
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
