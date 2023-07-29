from abc import abstractmethod
from dataclasses import dataclass

from ..animations.animation.conditions.condition import Condition
from .events import (
    Event,
    KeyPress,
    KeyRelease,
    MouseDrag,
    MouseMotion,
    MousePress,
    MouseRelease,
    MouseScroll
)
from .toplevel import Toplevel


class EventCapturer(Condition):
    __slots__ = ()

    @abstractmethod
    def capture(
        self,
        event: Event
    ) -> bool:
        pass

    def judge(self) -> bool:
        event_queue = Toplevel.event_queue
        for event in event_queue:
            if self.capture(event):
                event_queue.remove(event)
                Toplevel._event = event
                return True
        return False

    @classmethod
    def _match_symbol(
        cls,
        required_symbol: int | None,
        symbol: int
    ) -> bool:
        return required_symbol is None or symbol == required_symbol

    @classmethod
    def _match_modifiers(
        cls,
        required_modifiers: int | None,
        modifiers: int
    ) -> bool:
        return required_modifiers is None or (required_modifiers & modifiers == required_modifiers)


@dataclass(
    frozen=True,
    slots=True
)
class KeyPressCaptured(EventCapturer):
    symbol: int | None = None
    modifiers: int | None = None

    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, KeyPress) \
            and self._match_symbol(self.symbol, event.symbol) \
            and self._match_modifiers(self.modifiers, event.modifiers)


@dataclass(
    frozen=True,
    slots=True
)
class KeyReleaseCaptured(EventCapturer):
    symbol: int | None = None
    modifiers: int | None = None

    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, KeyRelease) \
            and self._match_symbol(self.symbol, event.symbol) \
            and self._match_modifiers(self.modifiers, event.modifiers)


@dataclass(
    frozen=True,
    slots=True
)
class MouseMotionCaptured(EventCapturer):
    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseMotion)


@dataclass(
    frozen=True,
    slots=True
)
class MouseDragCaptured(EventCapturer):
    buttons: int | None = None
    modifiers: int | None = None

    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseDrag) \
            and self._match_modifiers(self.buttons, event.buttons) \
            and self._match_modifiers(self.modifiers, event.modifiers)


@dataclass(
    frozen=True,
    slots=True
)
class MousePressCaptured(EventCapturer):
    buttons: int | None = None
    modifiers: int | None = None

    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MousePress) \
            and self._match_modifiers(self.buttons, event.buttons) \
            and self._match_modifiers(self.modifiers, event.modifiers)


@dataclass(
    frozen=True,
    slots=True
)
class MouseReleaseCaptured(EventCapturer):
    buttons: int | None = None
    modifiers: int | None = None

    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseRelease) \
            and self._match_modifiers(self.buttons, event.buttons) \
            and self._match_modifiers(self.modifiers, event.modifiers)


@dataclass(
    frozen=True,
    slots=True
)
class MouseScrollCaptured(EventCapturer):
    def capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseScroll)
