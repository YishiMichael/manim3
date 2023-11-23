from __future__ import annotations


from typing import (
    Iterator,
    Self
)

import moderngl
import pyglet
import pyglet.gl as gl

from .event import Event
from .events import Events
from .toplevel import Toplevel
from .toplevel_resource import ToplevelResource


class WindowHandlers:
    __slots__ = ("__weakref__",)

    def on_key_press(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.key_press(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_key_release(
        self: Self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel._get_window().push_event(Events.key_release(
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
        Toplevel._get_window().push_event(Events.mouse_motion(
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
        Toplevel._get_window().push_event(Events.mouse_drag(
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
        Toplevel._get_window().push_event(Events.mouse_press(
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
        Toplevel._get_window().push_event(Events.mouse_release(
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
        Toplevel._get_window().push_event(Events.mouse_scroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        ))

    def on_close(
        self: Self
    ) -> None:
        Toplevel._get_window()._pyglet_window.close()


class Window(ToplevelResource):
    __slots__ = (
        "_pyglet_window",
        "_event_queue",
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
        self._event_queue: list[Event] = []
        # Keep a strong reference to the handler object, as per
        # `https://pyglet.readthedocs.io/en/latest/programming_guide/events.html#stacking-event-handlers`.
        self._window_handlers: WindowHandlers = window_handlers

    def __contextmanager__(
        self: Self
    ) -> Iterator[None]:
        Toplevel._window = self
        yield
        self._pyglet_window.close()
        Toplevel._window = None

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

    def update_frame(
        self: Self,
        framebuffer: moderngl.Framebuffer
    ) -> None:
        src = framebuffer
        dst = Toplevel._get_context().screen_framebuffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, src.glo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, dst.glo)
        gl.glBlitFramebuffer(
            *src.viewport, *dst.viewport,
            gl.GL_COLOR_BUFFER_BIT, gl.GL_LINEAR
        )
        self._pyglet_window.flip()
