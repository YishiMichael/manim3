from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator
)

#from .events.key_press import KeyPress
#from .events.key_release import KeyRelease
#from .events.mouse_drag import MouseDrag
#from .events.mouse_motion import MouseMotion
#from .events.mouse_press import MousePress
#from .events.mouse_release import MouseRelease
#from .events.mouse_scroll import MouseScroll
from .config import Config

if TYPE_CHECKING:
    from .context import Context
    #from .events.event import Event
    from .scene import Scene
    from .window import Window


class Toplevel:
    __slots__ = ()

    _config: ClassVar[Config | None] = None
    #_event_queue: "ClassVar[list[Event] | None]" = None
    #_event: "ClassVar[Event | None]" = None
    _window: "ClassVar[Window | None]" = None
    _context: "ClassVar[Context | None]" = None
    _scene: "ClassVar[Scene | None]" = None

    @classmethod
    @property
    def config(cls) -> Config:
        assert (config := cls._config) is not None
        return config

    #@classmethod
    #@property
    #def event_queue(cls) -> "list[Event]":
    #    assert (event_queue := cls._event_queue) is not None
    #    return event_queue

    #@classmethod
    #@property
    #def event(cls) -> "Event":
    #    assert (event := cls._event) is not None
    #    return event

    @classmethod
    @property
    def window(cls) -> "Window":
        assert (window := cls._window) is not None
        return window

    @classmethod
    @property
    def context(cls) -> "Context":
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
        from .context import Context
        from .window import Window

        cls._config = config
        #cls._event_queue = []
        with Window(
            window_pixel_size=config.window_pixel_size,
            gl_version=config.gl_version,
            preview=config.preview
        ) as window:
            cls._window = window
            with Context(
                gl_version_code=config.gl_version_code,
                preview=config.preview
            ) as context:
                cls._context = context
                cls._scene = scene = scene_cls()
                yield scene
                cls._scene = None
                cls._context = None
            cls._window = None
        #cls._event = None
        #cls._event_queue = None
        cls._config = None

    #@classmethod
    #@contextmanager
    #def setup_window(
    #    cls,
    #    config: Config
    #) -> "Iterator[Window | None]":
    #    Window(
    #        window_pixel_size=config.window_pixel_size,
    #        gl_version=config.gl_version,
    #        preview=config.preview
    #    )
    #    if not config.preview:
    #        yield None
    #        return
    #    width, height = config.window_pixel_size
    #    major_version, minor_version = config.gl_version
    #    pyglet_window = Window(
    #        width=width,
    #        height=height,
    #        config=WindowConfig(
    #            double_buffer=True,
    #            major_version=major_version,
    #            minor_version=minor_version
    #        )
    #    )
    #    # Keep a strong reference to the handler object, as per
    #    # `https://pyglet.readthedocs.io/en/latest/programming_guide/events.html#stacking-event-handlers`.
    #    handlers = WindowHandlers()
    #    pyglet_window.push_handlers(handlers)
    #    yield pyglet_window
    #    pyglet_window.close()

    #@classmethod
    #@contextmanager
    #def setup_context(
    #    cls,
    #    config: Config
    #) -> "Iterator[Context]":
    #    context = Context(
    #        gl_version_code=config.gl_version_code,
    #        preview=config.preview
    #    )
    #    yield context
    #    context.release()
    #    #mgl_context = moderngl.create_context(
    #    #    require=config.gl_version_code,
    #    #    standalone=not config.preview
    #    #)
    #    #mgl_context.gc_mode = "auto"
    #    #yield Context(
    #    #    gl_version_code=config.gl_version_code,
    #    #    preview=config.preview
    #    #)
    #    #mgl_context.release()
