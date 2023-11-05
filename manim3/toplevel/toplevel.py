from __future__ import annotations


from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator,
    Self
)

from .config import Config

if TYPE_CHECKING:
    from .context import Context
    from .scene import Scene
    from .window import Window


class Toplevel:
    __slots__ = ()

    _config: ClassVar[Config | None] = None
    _window: ClassVar[Window | None] = None
    _context: ClassVar[Context | None] = None
    _scene: ClassVar[Scene | None] = None

    @classmethod
    @property
    def config(
        cls: type[Self]
    ) -> Config:
        assert (config := cls._config) is not None
        return config

    @classmethod
    @property
    def window(
        cls: type[Self]
    ) -> Window:
        assert (window := cls._window) is not None
        return window

    @classmethod
    @property
    def context(
        cls: type[Self]
    ) -> Context:
        assert (context := cls._context) is not None
        return context

    @classmethod
    @property
    def scene(
        cls: type[Self]
    ) -> Scene:
        assert (scene := cls._scene) is not None
        return scene

    @classmethod
    @contextmanager
    def configure(
        cls: type[Self],
        config: Config,
        scene_cls: type[Scene]
    ) -> Iterator[Scene]:
        cls._config = config
        with cls.setup_window(config) as window:
            cls._window = window
            with cls.setup_context(config) as context:
                cls._context = context
                cls._scene = scene = scene_cls()
                yield scene
                cls._scene = None
                cls._context = None
            cls._window = None
        cls._config = None

    @classmethod
    @contextmanager
    def setup_window(
        cls: type[Self],
        config: Config
    ) -> Iterator[Window | None]:
        from .window import Window
        window = Window(
            window_pixel_size=config.window_pixel_size,
            gl_version=config.gl_version,
            preview=config.preview
        )
        yield window
        window.close()

    @classmethod
    @contextmanager
    def setup_context(
        cls: type[Self],
        config: Config
    ) -> Iterator[Context | None]:
        from .context import Context
        context = Context(
            gl_version_code=config.gl_version_code,
            preview=config.preview
        )
        yield context
        context.release()
