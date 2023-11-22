from __future__ import annotations


from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Iterator,
    Self
)

if TYPE_CHECKING:
    from .config import Config
    from .context import Context
    from .logger import Logger
    from .renderer import Renderer
    from .scene import Scene
    from .window import Window


class Toplevel:
    __slots__ = ()

    _config: ClassVar[Config | None] = None
    _logger: ClassVar[Logger | None] = None
    _window: ClassVar[Window | None] = None
    _context: ClassVar[Context | None] = None
    _renderer: ClassVar[Renderer | None] = None
    _scene: ClassVar[Scene | None] = None

    @classmethod
    def _get_config(
        cls: type[Self]
    ) -> Config:
        assert (config := cls._config) is not None
        return config

    @classmethod
    def _get_logger(
        cls: type[Self]
    ) -> Logger:
        assert (logger := cls._logger) is not None
        return logger

    @classmethod
    def _get_window(
        cls: type[Self]
    ) -> Window:
        assert (window := cls._window) is not None
        return window

    @classmethod
    def _get_context(
        cls: type[Self]
    ) -> Context:
        assert (context := cls._context) is not None
        return context

    @classmethod
    def _get_renderer(
        cls: type[Self]
    ) -> Renderer:
        assert (renderer := cls._renderer) is not None
        return renderer

    @classmethod
    def _get_scene(
        cls: type[Self]
    ) -> Scene:
        assert (scene := cls._scene) is not None
        return scene

    @classmethod
    @contextmanager
    def configure(
        cls: type[Self],
        config: Config
        #scene_cls: type[Scene]
    ) -> Iterator[None]:
        from .logger import Logger
        from .window import Window
        from .context import Context
        from .renderer import Renderer
        with (
            config,
            Logger(),
            Window(),
            Context(),
            Renderer()
        ):
            yield
        #cls._config = config
        #with Logger() as logger:
        #    cls._logger = logger
        #cls._window = Window()
        #cls._context = Context()
        #cls._renderer = Renderer()
        #yield
        #cls._renderer.clear_pipes()
        #cls._renderer = None
        #cls._context.release()
        #cls._context = None
        #cls._window.close()
        #cls._window = None
        ##cls._logger.close()
        #cls._logger = None
        #cls._config = None

    #@classmethod
    #@contextmanager
    #def setup_window(
    #    cls: type[Self]
    #    #config: Config
    #) -> Iterator[Window]:
    #    from .window import Window
    #    window = Window(
    #        #gl_version=config.gl_version,
    #        #livestream_pixel_size=config.livestream_pixel_size,
    #        #verbose=config.verbose
    #    )
    #    yield window
    #    window.close()

    #@classmethod
    #@contextmanager
    #def setup_context(
    #    cls: type[Self]
    #    #config: Config
    #) -> Iterator[Context]:
    #    from .context import Context
    #    context = Context(
    #        #gl_version_code=config.gl_version_code
    #        #preview=config.preview
    #    )
    #    yield context
    #    context.release()

    #@classmethod
    #@contextmanager
    #def setup_renderer(
    #    cls: type[Self]
    #    #config: Config
    #) -> Iterator[Renderer]:
    #    from .renderer import Renderer
    #    yield Renderer()

    #@classmethod
    #@contextmanager
    #def _set_toplevel_scene(
    #    cls: type[Self],
    #    scene: Scene
    #) -> Iterator[None]:
    #    cls._scene = scene
    #    yield
    #    cls._scene = None

    @classmethod
    def start_livestream(
        cls: type[Self]
    ) -> None:
        cls._get_renderer().start_livestream()

    @classmethod
    def stop_livestream(
        cls: type[Self]
    ) -> None:
        cls._get_renderer().stop_livestream()

    @classmethod
    def start_recording(
        cls: type[Self],
        filename: str | None = None
    ) -> None:
        cls._get_renderer().start_recording(filename)

    @classmethod
    def stop_recording(
        cls: type[Self],
        filename: str | None = None
    ) -> None:
        cls._get_renderer().stop_recording(filename)

    @classmethod
    def snapshot(
        cls: type[Self],
        filename: str | None = None
    ) -> None:
        cls._get_renderer().snapshot(filename)

    @classmethod
    @contextmanager
    def livestream(
        cls: type[Self]
    ) -> Iterator[None]:
        cls.start_livestream()
        yield
        cls.stop_livestream()

    @classmethod
    @contextmanager
    def recording(
        cls: type[Self],
        filename: str | None = None
    ) -> Iterator[None]:
        cls.start_recording(filename)
        yield
        cls.stop_recording(filename)
