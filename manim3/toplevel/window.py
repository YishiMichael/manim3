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
        Toplevel.window.event_queue.append(Events.key_press(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_key_release(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel.window.event_queue.append(Events.key_release(
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
        Toplevel.window.event_queue.append(Events.mouse_motion(
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
        Toplevel.window.event_queue.append(Events.mouse_drag(
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
        Toplevel.window.event_queue.append(Events.mouse_press(
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
        Toplevel.window.event_queue.append(Events.mouse_release(
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
        Toplevel.window.event_queue.append(Events.mouse_scroll(
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
        "_pyglet_window",
        "_event_queue",
        "_recent_event"
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
            # Keep a strong reference to the handler object, as per
            # `https://pyglet.readthedocs.io/en/latest/programming_guide/events.html#stacking-event-handlers`.
            handlers = WindowHandlers()
            pyglet_window.push_handlers(handlers)
        else:
            pyglet_window = None

        self._pyglet_window: PygletWindow | None = pyglet_window
        self._event_queue: list[Event] = []
        self._recent_event: Event | None = None

    #def __enter__(self):
    #    return self

    #def __exit__(
    #    self,
    #    *args,
    #    **kwargs
    #) -> None:
    #    self.close()

    @property
    def pyglet_window(
        self: Self
    ) -> PygletWindow:
        assert (pyglet_window := self._pyglet_window) is not None
        return pyglet_window

    @property
    def event_queue(
        self: Self
    ) -> list[Event]:
        return self._event_queue

    @property
    def recent_event(
        self: Self
    ) -> Event:
        assert (recent_event := self._recent_event) is not None
        return recent_event

    def capture_event(
        self: Self,
        targeting_event: Event
    ) -> bool:
        event_queue = self.event_queue
        for event in event_queue:
            if targeting_event._capture(event):
                event_queue.remove(event)
                self._recent_event = event
                return True
        return False

    def close(
        self: Self
    ) -> None:
        if (pyglet_window := self._pyglet_window) is not None:
            pyglet_window.close()
