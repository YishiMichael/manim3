from __future__ import annotations


from typing import (
    Iterator,
    Self
)

import pyglet

from .events import (
    Event,
    EventInfo,
    KeyPress,
    KeyRelease,
    MouseDrag,
    MouseMotion,
    MousePress,
    MouseRelease,
    MouseScroll
)
from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class WindowHandlers:
    __slots__ = ("__weakref__",)

    def on_key_press(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event_info(KeyPress(
            symbol=symbol,
            modifiers=modifiers
        )._event_info)

    def on_key_release(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event_info(KeyRelease(
            symbol=symbol,
            modifiers=modifiers
        )._event_info)

    def on_mouse_motion(
        self: Self,
        x: int,
        y: int,
        dx: int,
        dy: int
    ) -> None:
        Toplevel._get_window().push_event_info(MouseMotion(
            x=x,
            y=y,
            dx=dx,
            dy=dy
        )._event_info)

    def on_mouse_drag(
        self: Self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event_info(MouseDrag(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            buttons=buttons,
            modifiers=modifiers
        )._event_info)

    def on_mouse_press(
        self: Self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event_info(MousePress(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        )._event_info)

    def on_mouse_release(
        self: Self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event_info(MouseRelease(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        )._event_info)

    def on_mouse_scroll(
        self: Self,
        x: int,
        y: int,
        scroll_x: float,
        scroll_y: float
    ) -> None:
        Toplevel._get_window().push_event_info(MouseScroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        )._event_info)

    def on_close(
        self: Self
    ) -> None:
        Toplevel._get_window()._pyglet_window.close()


class Window(ToplevelResource):
    __slots__ = (
        "_pyglet_window",
        "_event_info_queue",
        "_window_handlers"
    )

    def __init__(
        self: Self
    ) -> None:
        super().__init__()
        window_width, window_height = Toplevel._get_config().window_pixel_size
        major_version, minor_version = Toplevel._get_config().gl_version
        pyglet_window = pyglet.window.Window(
            width=window_width,
            height=window_height,
            config=pyglet.gl.Config(
                double_buffer=True,
                major_version=major_version,
                minor_version=minor_version
            )
        )
        window_handlers = WindowHandlers()
        pyglet_window.push_handlers(window_handlers)

        self._pyglet_window: pyglet.window.Window = pyglet_window
        self._event_info_queue: list[EventInfo] = []
        # Keep a strong reference to the handler object, according to
        # `https://pyglet.readthedocs.io/en/latest/programming_guide/events.html#stacking-event-handlers`.
        self._window_handlers: WindowHandlers = window_handlers

    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        Toplevel._window = self
        yield
        self._pyglet_window.close()
        Toplevel._window = None

    def push_event_info(
        self: Self,
        event_info: EventInfo
    ) -> None:
        self._event_info_queue.append(event_info)

    def capture_event_info_by(
        self: Self,
        event: Event
    ) -> bool:
        event_info_queue = self._event_info_queue
        for event_info in event_info_queue:
            if event._capture(event_info):
                event_info_queue.remove(event_info)
                return True
        return False

    def clear_event_info_queue(
        self: Self
    ) -> None:
        self._event_info_queue.clear()
