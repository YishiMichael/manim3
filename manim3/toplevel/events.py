from __future__ import annotations


from typing import Self

import attrs

from .toplevel import Toplevel


@attrs.frozen(kw_only=True)
class EventInfo:
    event_type: str
    symbol: int | None
    modifiers: int | None
    buttons: int | None
    x: int | None
    y: int | None
    dx: int | None
    dy: int | None
    scroll_x: float | None
    scroll_y: float | None


class Event:
    __slots__ = (
        "_event_info",
        "_captured_event_info"
    )

    def __init__(
        self: Self,
        *,
        event_type: str,
        symbol: int | None = None,
        modifiers: int | None = None,
        buttons: int | None = None,
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None,
        scroll_x: float | None = None,
        scroll_y: float | None = None
    ) -> None:
        super().__init__()
        self._event_info: EventInfo = EventInfo(
            event_type=event_type,
            symbol=symbol,
            modifiers=modifiers,
            buttons=buttons,
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        )
        self._captured_event_info: EventInfo | None = None

    def _capture(
        self: Self,
        event_info: EventInfo
    ) -> bool:
        target = self._event_info
        if (
            target.event_type == event_info.event_type
            and all(
                target_value is None or source_value is None
                or target_value == (target_value & source_value if masked else source_value)
                for target_value, source_value, masked in (
                    (target.symbol, event_info.symbol, False),
                    (target.modifiers, event_info.modifiers, True),
                    (target.buttons, event_info.buttons, True),
                )
            )
        ):
            self._captured_event_info = event_info
            return True
        return False

    def captured(
        self: Self
    ) -> bool:
        return self._captured_event_info is not None or Toplevel._get_window().capture_event_info_by(self)

    def get_captured_event_info(
        self: Self
    ) -> EventInfo | None:
        return self._captured_event_info


class KeyPress(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        symbol: int | None = None,
        modifiers: int | None = None
    ) -> None:
        super().__init__(
            event_type="key_press",
            symbol=symbol,
            modifiers=modifiers
        )


class KeyRelease(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        symbol: int | None = None,
        modifiers: int | None = None
    ) -> None:
        super().__init__(
            event_type="key_release",
            symbol=symbol,
            modifiers=modifiers
        )


class MouseMotion(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None
    ) -> None:
        super().__init__(
            event_type="mouse_motion",
            x=x,
            y=y,
            dx=dx,
            dy=dy
        )


class MouseDrag(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        x: int | None = None,
        y: int | None = None,
        dx: int | None = None,
        dy: int | None = None,
        buttons: int | None = None,
        modifiers: int | None = None
    ) -> None:
        super().__init__(
            event_type="mouse_drag",
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            buttons=buttons,
            modifiers=modifiers
        )


class MousePress(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        x: int | None = None,
        y: int | None = None,
        buttons: int | None = None,
        modifiers: int | None = None
    ) -> None:
        super().__init__(
            event_type="mouse_press",
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        )


class MouseRelease(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        x: int | None = None,
        y: int | None = None,
        buttons: int | None = None,
        modifiers: int | None = None
    ) -> None:
        super().__init__(
            event_type="mouse_release",
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        )


class MouseScroll(Event):
    __slots__ = ()

    def __init__(
        self: Self,
        x: int | None = None,
        y: int | None = None,
        scroll_x: float | None = None,
        scroll_y: float | None = None
    ) -> None:
        super().__init__(
            event_type="mouse_scroll",
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        )
