from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator
)

import moderngl
from pyglet.gl import Config as WindowConfig
from pyglet.window import Window

from .config import Config
from .context import Context
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

if TYPE_CHECKING:
    from .scene import Scene


class WindowHandlers:
    __slots__ = ("__weakref__",)

    def on_key_press(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel.event_queue.append(KeyPress(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_key_release(
        self,
        symbol: int,
        modifiers: int
    ) -> None:
        Toplevel.event_queue.append(KeyRelease(
            symbol=symbol,
            modifiers=modifiers
        ))

    def on_mouse_motion(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int
    ) -> None:
        Toplevel.event_queue.append(MouseMotion(
            x=x,
            y=y,
            dx=dx,
            dy=dy
        ))

    def on_mouse_drag(
        self,
        x: int,
        y: int,
        dx: int,
        dy: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel.event_queue.append(MouseDrag(
            x=x,
            y=y,
            dx=dx,
            dy=dy,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_press(
        self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel.event_queue.append(MousePress(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_release(
        self,
        x: int,
        y: int,
        buttons: int,
        modifiers: int
    ) -> None:
        Toplevel.event_queue.append(MouseRelease(
            x=x,
            y=y,
            buttons=buttons,
            modifiers=modifiers
        ))

    def on_mouse_scroll(
        self,
        x: int,
        y: int,
        scroll_x: float,
        scroll_y: float
    ) -> None:
        Toplevel.event_queue.append(MouseScroll(
            x=x,
            y=y,
            scroll_x=scroll_x,
            scroll_y=scroll_y
        ))

    def on_close(self) -> None:
        Toplevel.window.close()


class Toplevel:
    __slots__ = ()

    _config: ClassVar[Config | None] = None
    _event_queue: ClassVar[list[Event] | None] = None
    _event: ClassVar[Event | None] = None
    _window: ClassVar[Window | None] = None
    _context: ClassVar[Context | None] = None
    _scene: "ClassVar[Scene | None]" = None

    @classmethod
    @property
    def config(cls) -> Config:
        assert (config := cls._config) is not None
        return config

    @classmethod
    @property
    def event_queue(cls) -> list[Event]:
        assert (event_queue := cls._event_queue) is not None
        return event_queue

    @classmethod
    @property
    def event(cls) -> Event:
        assert (event := cls._event) is not None
        return event

    @classmethod
    @property
    def window(cls) -> Window:
        assert (window := cls._window) is not None
        return window

    @classmethod
    @property
    def context(cls) -> Context:
        assert (context := cls._context) is not None
        return context

    @classmethod
    @property
    def scene(cls) -> "Scene":
        assert (scene := cls._scene) is not None
        return scene

    @classmethod
    @contextmanager
    def configure(
        cls,
        config: Config,
        scene_cls: "type[Scene]"
    ) -> "Iterator[Scene]":
        cls._config = config
        cls._event_queue = []
        with cls.setup_window(config) as window:
            cls._window = window
            with cls.setup_context(config) as context:
                cls._context = context
                cls._scene = scene = scene_cls()
                yield scene
                cls._scene = None
                cls._context = None
            cls._window = None
        cls._event = None
        cls._event_queue = None
        cls._config = None

    @classmethod
    @contextmanager
    def setup_window(
        cls,
        config: Config
    ) -> Iterator[Window | None]:
        if not config.preview:
            yield None
            return
        width, height = config.window_pixel_size
        major_version, minor_version = config.gl_version
        pyglet_window = Window(
            width=width,
            height=height,
            config=WindowConfig(
                double_buffer=True,
                major_version=major_version,
                minor_version=minor_version
            )
        )
        handlers = WindowHandlers()
        pyglet_window.push_handlers(handlers)
        yield pyglet_window
        pyglet_window.close()

    @classmethod
    @contextmanager
    def setup_context(
        cls,
        config: Config
    ) -> Iterator[Context]:
        mgl_context = moderngl.create_context(
            require=config.gl_version_code,
            standalone=not config.preview
        )
        yield Context(
            mgl_context=mgl_context
        )
        mgl_context.release()
