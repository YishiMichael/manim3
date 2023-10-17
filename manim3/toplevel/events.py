from __future__ import annotations


from typing import (
    Never,
    Self
)

import attrs

from .event import Event


@attrs.frozen(kw_only=True)
class KeyPressEvent(Event):
    symbol: int | None
    modifiers: int | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, KeyPressEvent)
            and self._match(self.symbol, event.symbol, masked=False)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@attrs.frozen(kw_only=True)
class KeyReleaseEvent(Event):
    symbol: int | None
    modifiers: int | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, KeyReleaseEvent)
            and self._match(self.symbol, event.symbol, masked=False)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@attrs.frozen(kw_only=True)
class MouseMotionEvent(Event):
    x: int | None
    y: int | None
    dx: int | None
    dy: int | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseMotionEvent)


@attrs.frozen(kw_only=True)
class MouseDragEvent(Event):
    buttons: int | None
    modifiers: int | None
    x: int | None
    y: int | None
    dx: int | None
    dy: int | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MouseDragEvent)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@attrs.frozen(kw_only=True)
class MousePressEvent(Event):
    buttons: int | None
    modifiers: int | None
    x: int | None
    y: int | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MousePressEvent)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@attrs.frozen(kw_only=True)
class MouseReleaseEvent(Event):
    buttons: int | None
    modifiers: int | None
    x: int | None
    y: int | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MouseReleaseEvent)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@attrs.frozen(kw_only=True)
class MouseScrollEvent(Event):
    x: int | None
    y: int | None
    scroll_x: float | None
    scroll_y: float | None

    def _capture(
        self: Self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseScrollEvent)


class Events:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def key_press(
        cls: type[Self],
        symbol: int | None = None,
        modifiers: int | None = None
    ) -> KeyPressEvent:
        return KeyPressEvent(
            symbol=symbol,
            modifiers=modifiers
        )

    @classmethod
    def key_release(
        cls: type[Self],
        symbol: int | None = None,
        modifiers: int | None = None
    ) -> KeyReleaseEvent:
        return KeyReleaseEvent(
            symbol=symbol,
            modifiers=modifiers
        )

    @classmethod
    def mouse_motion(
        cls: type[Self],
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None
    ) -> MouseMotionEvent:
        return MouseMotionEvent(
            x=x,
            y=y,
            dx=dx,
            dy=dy
        )

    @classmethod
    def mouse_drag(
        cls: type[Self],
        buttons: int | None = None,
        modifiers: int | None = None,
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None
    ) -> MouseDragEvent:
        return MouseDragEvent(
            buttons=buttons,
            modifiers=modifiers,
            x=x,
            y=y,
            dx=dx,
            dy=dy
        )

    @classmethod
    def mouse_press(
        cls: type[Self],
        buttons: int | None = None,
        modifiers: int | None = None,
        x: int | None = None,
        y: int | None = None
    ) -> MousePressEvent:
        return MousePressEvent(
            buttons=buttons,
            modifiers=modifiers,
            x=x,
            y=y
        )

    @classmethod
    def mouse_release(
        cls: type[Self],
        buttons: int | None = None,
        modifiers: int | None = None,
        x: int | None = None,
        y: int | None = None
    ) -> MouseReleaseEvent:
        return MouseReleaseEvent(
            buttons=buttons,
            modifiers=modifiers,
            x=x,
            y=y
        )

    @classmethod
    def mouse_scroll(
        cls: type[Self],
        x: int | None = None,
        y: int | None = None,
        scroll_x: float | None = None,
        scroll_y: float | None = None
    ) -> MouseScrollEvent:
        return MouseScrollEvent(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        )
