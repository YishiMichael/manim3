from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator
)

from .config import Config
from .context import Context
from .window import Window

if TYPE_CHECKING:
    from .scene import Scene


class Toplevel:
    __slots__ = ()

    _config: ClassVar[Config | None] = None
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
        with Window.get_window(config) as window:
            cls._window = window
            with Context.get_context(config) as context:
                cls._context = context
                cls._scene = scene = scene_cls()
                yield scene
                cls._scene = None
                cls._context = None
            cls._window = None
        cls._config = None
