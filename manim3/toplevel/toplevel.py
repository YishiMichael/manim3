from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator
)

from .config import Config

if TYPE_CHECKING:
    from .context import Context
    from .scene import Scene
    from .window import Window


class Toplevel:
    __slots__ = ()

    _GL_VERSION: ClassVar[tuple[int, int]] = (4, 3)
    _GL_VERSION_CODE: ClassVar[int] = 430

    #_scene_name: ClassVar[str | None] = None
    _config: ClassVar[Config | None] = None
    _window: "ClassVar[Window | None]" = None
    _context: "ClassVar[Context | None]" = None
    _scene: "ClassVar[Scene | None]" = None

    #@classmethod
    #@contextmanager
    #def _assign_scene_name(
    #    cls,
    #    scene_name: str
    #) -> Iterator[str]:
    #    cls._scene_name = scene_name
    #    yield scene_name
    #    cls._scene_name = None

    #@classmethod
    #@contextmanager
    #def _assign_config(
    #    cls,
    #    config: Config
    #) -> Iterator[Config]:
    #    cls._config = config
    #    yield config
    #    cls._config = None

    #@classmethod
    #@contextmanager
    #def _assign_context(
    #    cls,
    #    context: "Context"
    #) -> "Iterator[Context]":
    #    cls._context = context
    #    yield context
    #    cls._context = None

    #@classmethod
    #@contextmanager
    #def _assign_scene(
    #    cls,
    #    scene: "Scene"
    #) -> "Iterator[Scene]":
    #    cls._scene = scene
    #    yield scene
    #    cls._scene = None

    #@classmethod
    #@property
    #def scene_name(cls) -> str:
    #    assert (scene_name := cls._scene_name) is not None
    #    return scene_name

    @classmethod
    @property
    def config(cls) -> Config:
        assert (config := cls._config) is not None
        return config

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
        #cls._scene_name = scene_cls.__name__
        cls._config = config
        with Window.get_window() as window:
            cls._window = window
            with Context.get_context() as context:
                cls._context = context
                cls._scene = scene = scene_cls()
                yield scene
                cls._scene = None
                cls._context = None
            cls._window = None
        cls._config = None
        #cls._scene_name = None
