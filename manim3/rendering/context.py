__all__ = ["ContextSingleton"]


import atexit
from typing import ClassVar

import moderngl
from moderngl_window.context.pyglet.window import Window

from ..rendering.config import ConfigSingleton


class ContextSingleton:
    __slots__ = ()

    _INSTANCE: ClassVar[moderngl.Context | None] = None
    _WINDOW: ClassVar[Window | None] = None
    _WINDOW_FRAMEBUFFER: ClassVar[moderngl.Framebuffer | None] = None

    def __new__(cls) -> moderngl.Context:
        if cls._INSTANCE is not None:
            return cls._INSTANCE
        if ConfigSingleton().preview:
            window = Window(
                size=ConfigSingleton().window_pixel_size,
                fullscreen=False,
                resizable=True,
                gl_version=(3, 3),
                vsync=True,
                cursor=True
            )
            context = window.ctx
            window_framebuffer = context.detect_framebuffer()
        else:
            window = None
            context = moderngl.create_context(standalone=True)
            window_framebuffer = None
        atexit.register(lambda: context.release())
        cls._INSTANCE = context
        cls._WINDOW = window
        cls._WINDOW_FRAMEBUFFER = window_framebuffer
        return context
