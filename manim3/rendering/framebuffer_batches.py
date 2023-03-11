__all__ = [
    "ColorFramebufferBatch",
    "SceneFramebufferBatch",
    "SimpleFramebufferBatch"
]


from dataclasses import dataclass

import moderngl

from ..rendering.framebuffer_batch import FramebufferBatch


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class SimpleFramebufferBatchStruct:
    color_texture: moderngl.Texture
    depth_texture: moderngl.Texture
    framebuffer: moderngl.Framebuffer


class SimpleFramebufferBatch(FramebufferBatch[SimpleFramebufferBatchStruct]):
    __slots__ = ()

    @classmethod
    def _new_instance(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        samples: int,
        dtype: str
    ) -> SimpleFramebufferBatchStruct:
        color_texture = cls.texture(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )
        depth_texture = cls.depth_texture(
            size=size,
            samples=samples
        )
        framebuffer = cls.framebuffer(
            color_attachments=[color_texture],
            depth_attachment=depth_texture
        )
        return SimpleFramebufferBatchStruct(
            color_texture=color_texture,
            depth_texture=depth_texture,
            framebuffer=framebuffer
        )

    @classmethod
    def _init_instance(
        cls,
        instance: SimpleFramebufferBatchStruct
    ) -> None:
        instance.framebuffer.clear()


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class ColorFramebufferBatchStruct:
    color_texture: moderngl.Texture
    framebuffer: moderngl.Framebuffer


class ColorFramebufferBatch(FramebufferBatch[ColorFramebufferBatchStruct]):
    __slots__ = ()

    @classmethod
    def _new_instance(
        cls,
        *,
        size: tuple[int, int],
        components: int,
        samples: int,
        dtype: str
    ) -> ColorFramebufferBatchStruct:
        color_texture = cls.texture(
            size=size,
            components=components,
            samples=samples,
            dtype=dtype
        )
        framebuffer = cls.framebuffer(
            color_attachments=[color_texture],
            depth_attachment=None
        )
        return ColorFramebufferBatchStruct(
            color_texture=color_texture,
            framebuffer=framebuffer
        )

    @classmethod
    def _init_instance(
        cls,
        instance: ColorFramebufferBatchStruct
    ) -> None:
        instance.framebuffer.clear()


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class SceneFramebufferBatchStruct:
    opaque_texture: moderngl.Texture
    accum_texture: moderngl.Texture
    revealage_texture: moderngl.Texture
    depth_texture: moderngl.Texture
    opaque_framebuffer: moderngl.Framebuffer
    accum_framebuffer: moderngl.Framebuffer
    revealage_framebuffer: moderngl.Framebuffer


class SceneFramebufferBatch(FramebufferBatch[SceneFramebufferBatchStruct]):
    __slots__ = ()

    @classmethod
    def _new_instance(
        cls,
        *,
        size: tuple[int, int],
        components: int,  # rejected
        samples: int,
        dtype: str  # rejected
    ) -> SceneFramebufferBatchStruct:
        opaque_texture = cls.texture(
            size=size,
            components=4,
            samples=samples,
            dtype="f1"
        )
        accum_texture = cls.texture(
            size=size,
            components=4,
            samples=samples,
            dtype="f2"
        )
        revealage_texture = cls.texture(
            size=size,
            components=1,
            samples=samples,
            dtype="f1"
        )
        depth_texture = cls.depth_texture(
            size=size,
            samples=samples
        )
        opaque_framebuffer = cls.framebuffer(
            color_attachments=[opaque_texture],
            depth_attachment=depth_texture
        )
        accum_framebuffer = cls.framebuffer(
            color_attachments=[accum_texture],
            depth_attachment=depth_texture
        )
        revealage_framebuffer = cls.framebuffer(
            color_attachments=[revealage_texture],
            depth_attachment=depth_texture
        )
        return SceneFramebufferBatchStruct(
            opaque_texture=opaque_texture,
            accum_texture=accum_texture,
            revealage_texture=revealage_texture,
            depth_texture=depth_texture,
            opaque_framebuffer=opaque_framebuffer,
            accum_framebuffer=accum_framebuffer,
            revealage_framebuffer=revealage_framebuffer
        )

    @classmethod
    def _init_instance(
        cls,
        instance: SceneFramebufferBatchStruct
    ) -> None:
        instance.opaque_framebuffer.clear()
        instance.accum_framebuffer.clear()
        instance.revealage_framebuffer.clear(red=1.0)  # initialize `revealage` with 1.0
        # Test against each fragment by the depth buffer, but never write to it.
        instance.accum_framebuffer.depth_mask = False
        instance.revealage_framebuffer.depth_mask = False
