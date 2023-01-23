__all__ = ["ContextSingleton"]


import atexit

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow

from ..constants import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH
)


class ContextSingleton:
    _window: PygletWindow = PygletWindow(
        size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
        fullscreen=False,
        resizable=True,
        gl_version=(4, 3),
        vsync=True,
        cursor=True
    )
    _version_string: str = "#version 430 core"
    _INSTANCE: moderngl.Context = _window.ctx
    #_INSTANCE.gc_mode = "auto"
    atexit.register(lambda: ContextSingleton._INSTANCE.release())

    def __new__(cls):
        return cls._INSTANCE

    @classmethod
    def get_window(cls) -> PygletWindow:
        return cls._window

    @classmethod
    def get_version_string(cls) -> str:
        return cls._version_string
