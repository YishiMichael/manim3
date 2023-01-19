__all__ = ["ContextSingleton"]


import atexit

import moderngl
from moderngl_window.context.pyglet.window import Window as PygletWindow

from ..constants import (
    PIXEL_HEIGHT,
    PIXEL_WIDTH
)


class ContextSingleton:
    window = PygletWindow(
        size=(PIXEL_WIDTH // 2, PIXEL_HEIGHT // 2),  # TODO
        fullscreen=False,
        resizable=True,
        gl_version=(4, 3),
        vsync=True,
        cursor=True
    )
    _INSTANCE: moderngl.Context = window.ctx
    #_INSTANCE.gc_mode = "auto"
    atexit.register(lambda: ContextSingleton._INSTANCE.release())

    def __new__(cls):
        return cls._INSTANCE

    @classmethod
    def get_window(cls):
        return cls.window
