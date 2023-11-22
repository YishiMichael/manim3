from __future__ import annotations


import pathlib
from contextlib import contextmanager
from typing import (
    IO,
    TYPE_CHECKING,
    ClassVar,
    Iterator,
    Self
)

from .config import Config

if TYPE_CHECKING:
    from .context import Context
    from .renderer import Renderer
    from .scene import Scene
    from .window import Window


class Toplevel:
    __slots__ = ()

    _config: ClassVar[Config | None] = None
    _window: ClassVar[Window | None] = None
    _context: ClassVar[Context | None] = None
    _renderer: ClassVar[Renderer | None] = None
    _scene: ClassVar[Scene | None] = None
    _streaming: ClassVar[bool] = False
    _video_process_items: ClassVar[dict[pathlib.Path, tuple[IO[bytes], bool]]] = {}

    @classmethod
    def _get_config(
        cls: type[Self]
    ) -> Config:
        assert (config := cls._config) is not None
        return config

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
        from .window import Window
        from .context import Context
        from .renderer import Renderer

        cls._config = config
        cls._window = Window()
        cls._context = Context()
        cls._renderer = Renderer()
        yield
        cls._renderer = None
        cls._context.release()
        cls._context = None
        cls._window.close()
        cls._window = None
        cls._config = None

    #@classmethod
    #@contextmanager
    #def setup_window(
    #    cls: type[Self]
    #    #config: Config
    #) -> Iterator[Window]:
    #    from .window import Window
    #    window = Window(
    #        #gl_version=config.gl_version,
    #        #streaming_pixel_size=config.streaming_pixel_size,
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

    @classmethod
    @contextmanager
    def _set_toplevel_scene(
        cls: type[Self],
        scene: Scene
    ) -> Iterator[None]:
        cls._scene = scene
        yield
        cls._scene = None

    @classmethod
    def start_streaming(
        cls: type[Self]
    ) -> None:
        cls._get_renderer().start_streaming()

    @classmethod
    def stop_streaming(
        cls: type[Self]
    ) -> None:
        cls._get_renderer().stop_streaming()

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
    def screenshot(
        cls: type[Self],
        filename: str | None = None
    ) -> None:
        cls._get_renderer().screenshot(filename)

    @classmethod
    @contextmanager
    def streaming(
        cls: type[Self]
    ) -> Iterator[None]:
        cls.start_streaming()
        yield
        cls.stop_streaming()

    @classmethod
    @contextmanager
    def recording(
        cls: type[Self],
        filename: str | None = None
    ) -> Iterator[None]:
        cls.start_recording(filename)
        yield
        cls.stop_recording(filename)
