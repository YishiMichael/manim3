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


class TemporaryResource(Generic[_T], ABC):
    __slots__ = (
        "_parameters",
        "_instance"
    )

    _VACANT_INSTANCES: dict[tuple, list[_T]]

    def __init_subclass__(cls) -> None:
        cls._VACANT_INSTANCES = {}

    def __init__(
        self,
        *args,
        **kwargs
    ) -> None:
        parameters = (*args, *kwargs.values())
        if (vacant_instances := self._VACANT_INSTANCES.get(parameters)) is not None and vacant_instances:
            instance = vacant_instances.pop()
        else:
            instance = self._new_instance(*args, **kwargs)
        self._init_instance(instance)
        self._parameters: tuple = parameters
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
        *args,
        **kwargs
    ) -> _T:
        pass

    @classmethod
    @abstractmethod
    def _init_instance(
        cls,
        instance: _T
    ) -> None:
        pass


class FramebufferBatch(TemporaryResource[_T]):
    __slots__ = ()

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
        super().__init__(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )

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

    @classmethod
    @abstractmethod
    def _new_instance(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        samples: int,
        dtype: str
    ) -> _T:
        pass
