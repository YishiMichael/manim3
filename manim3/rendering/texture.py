#from contextlib import contextmanager
#from typing import (
#    ClassVar,
#    Iterator
#)

#import moderngl

#from ..config import Config
#from .context import Context


#class TextureFactory:
#    __slots__ = ()

#    _VACANT_TEXTURES: ClassVar[dict[tuple[tuple[int, int], int, str], list[moderngl.Texture]]] = {}
#    _VACANT_DEPTH_TEXTURES: ClassVar[dict[tuple[tuple[int, int]], list[moderngl.Texture]]] = {}

#    def __new__(cls):
#        raise TypeError

#    @classmethod
#    @contextmanager
#    def texture(
#        cls,
#        *,
#        size: tuple[int, int] | None = None,
#        components: int,
#        samples: int | None = None,
#        dtype: str = "f1"
#    ) -> Iterator[moderngl.Texture]:
#        if size is None:
#            size = Config().size.pixel_size
#        if samples is None:
#            samples = Config().rendering.msaa_samples
#        parameters = (size, components, dtype)
#        vacant_list = cls._VACANT_TEXTURES.setdefault(parameters, [])
#        if vacant_list:
#            texture = vacant_list.pop()
#        else:
#            texture = Context.texture(
#                size=size,
#                components=components,
#                samples=samples,
#                dtype=dtype
#            )
#        yield texture
#        vacant_list.append(texture)

#    @classmethod
#    @contextmanager
#    def depth_texture(
#        cls,
#        *,
#        size: tuple[int, int] | None = None,
#        samples: int | None = None
#    ) -> Iterator[moderngl.Texture]:
#        if size is None:
#            size = Config().size.pixel_size
#        if samples is None:
#            samples = Config().rendering.msaa_samples
#        parameters = (size,)
#        vacant_list = cls._VACANT_DEPTH_TEXTURES.setdefault(parameters, [])
#        if vacant_list:
#            depth_texture = vacant_list.pop()
#        else:
#            depth_texture = Context.depth_texture(
#                size=size,
#                samples=samples
#            )
#        yield depth_texture
#        vacant_list.append(depth_texture)
