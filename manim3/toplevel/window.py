from __future__ import annotations


from typing import Self

from pyglet.gl import Config as PygletWindowConfig
from pyglet.window import Window as PygletWindow

from .event import Event
from .events import Events
from .toplevel import Toplevel


class WindowHandlers:
    __slots__ = ("__weakref__",)

    def on_key_press(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel.window.push_event(Events.key_press(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_key_release(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel.window.push_event(Events.key_release(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_mouse_motion(
        self: Self,
        x: int,
        y: int,
        dx: int,
        dy: int
    ) -> None:
        Toplevel.window.push_event(Events.mouse_motion(
            x=x,
            y=y,
            dx=dx,
            dy=dy
        ))

    def on_mouse_drag(
        self: Self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel.window.push_event(Events.mouse_drag(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_press(
        self: Self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel.window.push_event(Events.mouse_press(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_release(
        self: Self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel.window.push_event(Events.mouse_release(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_scroll(
        self: Self,
        x: int,
        y: int,
        scroll_x: float,
        scroll_y: float
    ) -> None:
        Toplevel.window.push_event(Events.mouse_scroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        ))

    def on_close(
        self: Self
    ) -> None:
        Toplevel.window.close()


class Window:
    __slots__ = (
        "_window_handlers",
        "_pyglet_window",
        "_event_queue"
    )

    def __init__(
        self: Self,
        window_pixel_size: tuple[int, int],
        gl_version: tuple[int, int],
        preview: bool
    ) -> None:
        super().__init__()
        if preview:
            width, height = window_pixel_size
            major_version, minor_version = gl_version
            pyglet_window = PygletWindow(
                width=width,
                height=height,
                config=PygletWindowConfig(
                    double_buffer=True,
                    major_version=major_version,
                    minor_version=minor_version
                )
            )
            window_handlers = WindowHandlers()
            pyglet_window.push_handlers(window_handlers)
        else:
            window_handlers = None
            pyglet_window = None

        # Keep a strong reference to the handler object, as per
        # `https://pyglet.readthedocs.io/en/latest/programming_guide/events.html#stacking-event-handlers`.
        self._window_handlers: WindowHandlers | None = window_handlers
        self._pyglet_window: PygletWindow | None = pyglet_window
        self._event_queue: list[Event] = []

    @property
    def pyglet_window(
        self: Self
    ) -> PygletWindow:
        assert (pyglet_window := self._pyglet_window) is not None
        return pyglet_window

    def push_event(
        self: Self,
        event: Event
    ) -> None:
        self._event_queue.append(event)

    def capture_event(
        self: Self,
        target_event: Event
    ) -> Event | None:
        event_queue = self._event_queue
        for event in event_queue:
            if target_event._capture(event):
                event_queue.remove(event)
                return event
        return None

    def clear_event_queue(
        self: Self
    ) -> None:
        self._event_queue.clear()

    def close(
        self: Self
    ) -> None:
        if (pyglet_window := self._pyglet_window) is not None:
            pyglet_window.close()
