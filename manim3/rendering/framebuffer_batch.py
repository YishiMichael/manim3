__all__ = ["FramebufferBatch"]


from abc import (
    ABC,
    abstractmethod
)
import atexit
from typing import (
    Generic,
    TypeVar
)

import moderngl

from ..rendering.config import ConfigSingleton
from ..rendering.context import ContextSingleton


_T = TypeVar("_T")
_ParamsT = TypeVar("_ParamsT", bound=tuple)


class TemporaryResource(Generic[_ParamsT, _T], ABC):
    _VACANT_INSTANCES: dict[_ParamsT, list[_T]]

    def __init_subclass__(cls) -> None:
        cls._VACANT_INSTANCES = {}

    def __init__(
        self,
        parameters: _ParamsT
    ):
        if (vacant_instances := self._VACANT_INSTANCES.get(parameters)) is not None and vacant_instances:
            instance = vacant_instances.pop()
        else:
            instance = self._new_instance(parameters)
        self._init_instance(instance)
        self._parameters: _ParamsT = parameters
        self._instance: _T = instance

    def __enter__(self) -> _T:
        return self._instance

    def __exit__(
        self,
        exc_type,
        exc_value,
        exc_traceback
    ) -> None:
        self._VACANT_INSTANCES.setdefault(self._parameters, []).append(self._instance)

    @classmethod
    @abstractmethod
    def _new_instance(
        cls,
        parameters: _ParamsT
    ) -> _T:
        pass

    @classmethod
    @abstractmethod
    def _init_instance(
        cls,
        instance: _T
    ) -> None:
        pass


# TODO: Why replacing `_ParamsT` with `tuple[tuple[int, int], int, int, str]` breaks the code?
class FramebufferBatch(TemporaryResource[_ParamsT, _T]):
    def __init__(
        self,
        *,
        size: tuple[int, int] | None = None,
        components: int = 4,
        samples: int = 0,
        dtype: str = "f1"
    ):
        if size is None:
            size = ConfigSingleton().pixel_size
        super().__init__((size, components, samples, dtype))

    @classmethod
    def texture(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        samples: int,
        dtype: str
    ) -> moderngl.Texture:
        texture = ContextSingleton().texture(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )
        atexit.register(lambda: texture.release())
        return texture

    @classmethod
    def depth_texture(
        cls,
        *,
        size: tuple[int, int],
        samples: int
    ) -> moderngl.Texture:
        depth_texture = ContextSingleton().depth_texture(
            size=size,
            samples=samples
        )
        atexit.register(lambda: depth_texture.release())
        return depth_texture

    @classmethod
    def framebuffer(
        cls,
        *,
        color_attachments: list[moderngl.Texture],
        depth_attachment: moderngl.Texture | None
    ) -> moderngl.Framebuffer:
        framebuffer = ContextSingleton().framebuffer(
            color_attachments=tuple(color_attachments),
            depth_attachment=depth_attachment
        )
        atexit.register(lambda: framebuffer.release())
        return framebuffer

    #@classmethod
    #def downsample_framebuffer(
    #    cls,
    #    src: moderngl.Framebuffer,
    #    dst: moderngl.Framebuffer
    #) -> None:
    #    ContextSingleton().copy_framebuffer(dst=dst, src=src)

    @classmethod
    def _new_instance(
        cls,
        parameters: tuple[tuple[int, int], int, int, str]
    ) -> _T:
        size, components, samples, dtype = parameters
        return cls._new_batch(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )

    @classmethod
    @abstractmethod
    def _new_batch(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        samples: int,
        dtype: str
    ) -> _T:
        pass
