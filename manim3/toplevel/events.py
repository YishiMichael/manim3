from dataclasses import dataclass

from .event import Event


@dataclass(
    frozen=True,
    slots=True
)
class KeyPress(Event):
    symbol: int | None
    modifiers: int | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, KeyPress)
            and self._match(self.symbol, event.symbol, masked=False)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@dataclass(
    frozen=True,
    slots=True
)
class KeyRelease(Event):
    symbol: int | None
    modifiers: int | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, KeyRelease)
            and self._match(self.symbol, event.symbol, masked=False)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@dataclass(
    frozen=True,
    slots=True
)
class MouseMotion(Event):
    x: int | None
    y: int | None
    dx: int | None
    dy: int | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseMotion)


@dataclass(
    frozen=True,
    slots=True
)
class MouseDrag(Event):
    buttons: int | None
    modifiers: int | None
    x: int | None
    y: int | None
    dx: int | None
    dy: int | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MouseDrag)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@dataclass(
    frozen=True,
    slots=True
)
class MousePress(Event):
    buttons: int | None
    modifiers: int | None
    x: int | None
    y: int | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MousePress)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@dataclass(
    frozen=True,
    slots=True
)
class MouseRelease(Event):
    buttons: int | None
    modifiers: int | None
    x: int | None
    y: int | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return (
            isinstance(event, MouseRelease)
            and self._match(self.buttons, event.buttons, masked=True)
            and self._match(self.modifiers, event.modifiers, masked=True)
        )


@dataclass(
    frozen=True,
    slots=True
)
class MouseScroll(Event):
    x: int | None
    y: int | None
    scroll_x: float | None
    scroll_y: float | None

    def _capture(
        self,
        event: Event
    ) -> bool:
        return isinstance(event, MouseScroll)


class Events:
    __slots__ = ()

    def __new__(cls):
        raise TypeError

    @classmethod
    def key_press(
        cls,
        symbol: int | None = None,
        modifiers: int | None = None
    ) -> KeyPress:
        return KeyPress(
            symbol=symbol,
            modifiers=modifiers
        )

    @classmethod
    def key_release(
        cls,
        symbol: int | None = None,
        modifiers: int | None = None
    ) -> KeyRelease:
        return KeyRelease(
            symbol=symbol,
            modifiers=modifiers
        )

    @classmethod
    def mouse_motion(
        cls,
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None
    ) -> MouseMotion:
        return MouseMotion(
            x=x,
            y=y,
            dx=dx,
            dy=dy
        )

    @classmethod
    def mouse_drag(
        cls,
        buttons: int | None = None,
        modifiers: int | None = None,
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None
    ) -> MouseDrag:
        return MouseDrag(
            buttons=buttons,
            modifiers=modifiers,
            x=x,
            y=y,
            dx=dx,
            dy=dy
        )

    @classmethod
    def mouse_press(
        cls,
        buttons: int | None = None,
        modifiers: int | None = None,
        x: int | None = None,
        y: int | None = None
    ) -> MousePress:
        return MousePress(
            buttons=buttons,
            modifiers=modifiers,
            x=x,
            y=y
        )

    @classmethod
    def mouse_release(
        cls,
        buttons: int | None = None,
        modifiers: int | None = None,
        x: int | None = None,
        y: int | None = None
    ) -> MouseRelease:
        return MouseRelease(
            buttons=buttons,
            modifiers=modifiers,
            x=x,
            y=y
        )

    @classmethod
    def mouse_scroll(
        cls,
        x: int | None = None,
        y: int | None = None,
        scroll_x: float | None = None,
        scroll_y: float | None = None
    ) -> MouseScroll:
        return MouseScroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        )
