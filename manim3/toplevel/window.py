from contextlib import contextmanager
from typing import Iterator

import pyglet

from .config import Config

#from pyglet.gl import Config as PygletConfig
#from pyglet.window import (
#    BaseWindow as PygletBaseWindow,
#    Window as PygletWindow
#)
#from pyglet.window.headless import HeadlessWindow as PygletHeadlessWindow

#from ..toplevel.toplevel import Toplevel


class Window:
    __slots__ = ("_pyglet_window",)

    def __init__(
        self,
        pyglet_window: pyglet.window.Window | None
    ) -> None:
        super().__init__()
        self._pyglet_window: pyglet.window.Window | None = pyglet_window

    @classmethod
    @contextmanager
    def get_window(
        cls,
        config: Config
    ) -> "Iterator[Window | None]":
        major_version, minor_version = config.gl_version
        #pyglet_config = pyglet.gl.Config(
        #    double_buffer=True,
        #    major_version=major_version,
        #    minor_version=minor_version
        #)
        #pyglet_window_cls = pyglet.window.Window if Toplevel.config.preview else pyglet.window.headless.HeadlessWindow
        if not config.preview:
            yield Window(pyglet_window=None)
            return
        width, height = config.window_pixel_size
        pyglet_window = pyglet.window.Window(
            width=width,
            height=height,
            config=pyglet.gl.Config(
                double_buffer=True,
                major_version=major_version,
                minor_version=minor_version
            )
        )
        yield Window(pyglet_window=pyglet_window)
        pyglet_window.close()
