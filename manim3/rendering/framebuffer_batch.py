__all__ = [
    "ColorFramebufferBatch",
    "SceneFramebufferBatch",
    "SimpleFramebufferBatch"
]


from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Generic,
    ParamSpec
)

import moderngl

from ..rendering.config import ConfigSingleton
from ..rendering.context import Context


_ResourceParameters = ParamSpec("_ResourceParameters")


class TemporaryResource(Generic[_ResourceParameters], ABC):
    __slots__ = ()

    _INSTANCE_TO_PARAMETERS_DICT: dict
    _VACANT_INSTANCES: dict[tuple, list]

    def __init_subclass__(cls) -> None:
        cls._INSTANCE_TO_PARAMETERS_DICT = {}
        cls._VACANT_INSTANCES = {}

    def __new__(
        cls,
        *args: _ResourceParameters.args,
        **kwargs: _ResourceParameters.kwargs
    ):
        parameters = (*args, *kwargs.values())
        if (vacant_instances := cls._VACANT_INSTANCES.get(parameters)) is not None and vacant_instances:
            self = vacant_instances.pop()
        else:
            self = super().__new__(cls)
            self._init_new_instance(*args, **kwargs)
            cls._INSTANCE_TO_PARAMETERS_DICT[self] = parameters
        #cls._init_instance(self)
        #self._parameters: tuple = parameters
        #self._instance: _T = instance
        return self

    def __init__(
        self,
        *args: _ResourceParameters.args,
        **kwargs: _ResourceParameters.kwargs
    ) -> None:
        self._init_instance()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        exc_traceback
    ) -> None:
        cls = self.__class__
        parameters = cls._INSTANCE_TO_PARAMETERS_DICT[self]
        cls._VACANT_INSTANCES.setdefault(parameters, []).append(self)

    @abstractmethod
    def _init_new_instance(
        self,
        *args: _ResourceParameters.args,
        **kwargs: _ResourceParameters.kwargs
    ) -> None:
        pass

    @abstractmethod
    def _init_instance(self) -> None:
        pass

    #def __init__(
    #    self,
    #    *,
    #    size: tuple[int, int] | None = None,
    #    components: int = 4,
    #    samples: int = 0,
    #    dtype: str = "f1"
    #):
    #    if size is None:
    #        size = ConfigSingleton().pixel_size
    #    super().__init__(
    #        size=size,
    #        components=components,
    #        samples=samples,
    #        dtype=dtype
    #    )

    #@classmethod
    #@abstractmethod
    #def _new_instance(
    #    cls,
    #    *,
    #    size: tuple[int, int],
    #    components: int,
    #    samples: int,
    #    dtype: str
    #) -> _T:
    #    pass


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class SimpleFramebufferBatchStruct:
#    color_texture: moderngl.Texture
#    depth_texture: moderngl.Texture
#    framebuffer: moderngl.Framebuffer


class SimpleFramebufferBatch(TemporaryResource):
    __slots__ = (
        "color_texture",
        "depth_texture",
        "framebuffer"
    )

    def _init_new_instance(
        self,
        size: tuple[int, int] | None = None,
        components: int = 4,
        samples: int = 0,
        dtype: str = "f1"
    ) -> None:
        if size is None:
            size = ConfigSingleton().pixel_size
        color_texture = Context.texture(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )
        depth_texture = Context.depth_texture(
            size=size,
            samples=samples
        )
        framebuffer = Context.framebuffer(
            color_attachments=(color_texture,),
            depth_attachment=depth_texture
        )
        self.color_texture: moderngl.Texture = color_texture
        self.depth_texture: moderngl.Texture = depth_texture
        self.framebuffer: moderngl.Framebuffer = framebuffer

    #@classmethod
    #def _new_instance(
    #    cls,
    #    *,
    #    size: tuple[int, int] | None = None,
    #    components: int = 4,
    #    samples: int = 0,
    #    dtype: str = "f1"
    #) -> SimpleFramebufferBatchStruct:
    #    if size is None:
    #        size = ConfigSingleton().pixel_size
    #    color_texture = Context.texture(
    #        size=size,
    #        components=components,
    #        samples=samples,
    #        dtype=dtype
    #    )
    #    depth_texture = Context.depth_texture(
    #        size=size,
    #        samples=samples
    #    )
    #    framebuffer = Context.framebuffer(
    #        color_attachments=(color_texture,),
    #        depth_attachment=depth_texture
    #    )
    #    return SimpleFramebufferBatchStruct(
    #        color_texture=color_texture,
    #        depth_texture=depth_texture,
    #        framebuffer=framebuffer
    #    )

    def _init_instance(self) -> None:
        self.framebuffer.clear()


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class ColorFramebufferBatchStruct:
#    color_texture: moderngl.Texture
#    framebuffer: moderngl.Framebuffer


class ColorFramebufferBatch(TemporaryResource):
    __slots__ = ()

    def _init_new_instance(
        self,
        *,
        size: tuple[int, int] | None = None,
        components: int = 4,
        samples: int = 0,
        dtype: str = "f1"
    ) -> None:
        if size is None:
            size = ConfigSingleton().pixel_size
        color_texture = Context.texture(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )
        framebuffer = Context.framebuffer(
            color_attachments=(color_texture,),
            depth_attachment=None
        )
        self.color_texture: moderngl.Texture = color_texture
        self.framebuffer: moderngl.Framebuffer = framebuffer

    #@classmethod
    #def _new_instance(
    #    cls,
    #    *,
    #    size: tuple[int, int] | None = None,
    #    components: int = 4,
    #    samples: int = 0,
    #    dtype: str = "f1"
    #) -> ColorFramebufferBatchStruct:
    #    if size is None:
    #        size = ConfigSingleton().pixel_size
    #    color_texture = Context.texture(
    #        size=size,
    #        components=components,
    #        samples=samples,
    #        dtype=dtype
    #    )
    #    framebuffer = Context.framebuffer(
    #        color_attachments=(color_texture,),
    #        depth_attachment=None
    #    )
    #    return ColorFramebufferBatchStruct(
    #        color_texture=color_texture,
    #        framebuffer=framebuffer
    #    )

    def _init_instance(self) -> None:
        self.framebuffer.clear()


#@dataclass(
#    frozen=True,
#    kw_only=True,
#    slots=True
#)
#class SceneFramebufferBatchStruct:
#    opaque_texture: moderngl.Texture
#    accum_texture: moderngl.Texture
#    revealage_texture: moderngl.Texture
#    depth_texture: moderngl.Texture
#    opaque_framebuffer: moderngl.Framebuffer
#    accum_framebuffer: moderngl.Framebuffer
#    revealage_framebuffer: moderngl.Framebuffer


class SceneFramebufferBatch(TemporaryResource):
    __slots__ = (
        "opaque_texture",
        "accum_texture",
        "revealage_texture",
        "depth_texture",
        "opaque_framebuffer",
        "accum_framebuffer",
        "revealage_framebuffer"
    )

    def _init_new_instance(
        self,
        *,
        size: tuple[int, int] | None = None,
        samples: int = 4
    ) -> None:
        if size is None:
            size = ConfigSingleton().pixel_size
        opaque_texture = Context.texture(
            size=size,
            components=4,
            samples=samples,
            dtype="f1"
        )
        accum_texture = Context.texture(
            size=size,
            components=4,
            samples=samples,
            dtype="f2"
        )
        revealage_texture = Context.texture(
            size=size,
            components=1,
            samples=samples,
            dtype="f1"
        )
        depth_texture = Context.depth_texture(
            size=size,
            samples=samples
        )
        opaque_framebuffer = Context.framebuffer(
            color_attachments=(opaque_texture,),
            depth_attachment=depth_texture
        )
        accum_framebuffer = Context.framebuffer(
            color_attachments=(accum_texture,),
            depth_attachment=depth_texture
        )
        revealage_framebuffer = Context.framebuffer(
            color_attachments=(revealage_texture,),
            depth_attachment=depth_texture
        )
        self.opaque_texture: moderngl.Texture = opaque_texture
        self.accum_texture: moderngl.Texture = accum_texture
        self.revealage_texture: moderngl.Texture = revealage_texture
        self.depth_texture: moderngl.Texture = depth_texture
        self.opaque_framebuffer: moderngl.Framebuffer = opaque_framebuffer
        self.accum_framebuffer: moderngl.Framebuffer = accum_framebuffer
        self.revealage_framebuffer: moderngl.Framebuffer = revealage_framebuffer
        #return SceneFramebufferBatchStruct(
        #    opaque_texture=opaque_texture,
        #    accum_texture=accum_texture,
        #    revealage_texture=revealage_texture,
        #    depth_texture=depth_texture,
        #    opaque_framebuffer=opaque_framebuffer,
        #    accum_framebuffer=accum_framebuffer,
        #    revealage_framebuffer=revealage_framebuffer
        #)

    def _init_instance(self) -> None:
        self.opaque_framebuffer.clear()
        self.accum_framebuffer.clear()
        self.revealage_framebuffer.clear(red=1.0)  # Tnitialize `revealage` with 1.0.
        # Test against each fragment by the depth buffer, but never write to it.
        self.accum_framebuffer.depth_mask = False
        self.revealage_framebuffer.depth_mask = False
