from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass

from ..animations.animation.condition import Condition
from .toplevel import Toplevel


@dataclass(
    frozen=True,
    slots=True
)
class Event(ABC):
    @abstractmethod
    def _capture(
        self,
        event: "Event"
    ) -> bool:
        pass

    @classmethod
    def _match(
        cls,
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

    def captured(self) -> "CapturedCondition":
        return CapturedCondition(self)


class CapturedCondition(Condition):
    __slots__ = ("_event",)

    def __init__(
        self,
        event: Event
    ) -> None:
        super().__init__()
        self._event: Event = event

    def judge(self) -> bool:
        return Toplevel.window.capture_event_by(self._event)
