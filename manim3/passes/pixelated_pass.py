__all__ = ["PixelatedPass"]


import moderngl
import numpy as np

from ..custom_typing import Real
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer_batches import ColorFramebufferBatch
from ..rendering.glsl_buffers import TextureStorage
from ..rendering.vertex_array import (
    ContextState,
    IndexedAttributesBuffer,
    VertexArray
)


class PixelatedPass(RenderPass):
    __slots__ = ()

    def __init__(
        self,
        pixelated_width: Real | None = None
    ) -> None:
        super().__init__()
        if pixelated_width is not None:
            self._pixelated_width_ = pixelated_width

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _pixelated_width_(cls) -> Real:
        return 0.1

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _color_map_(cls) -> moderngl.Texture:
        return NotImplemented

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_color_map_(
        cls,
        color_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_color_map",
            texture_array=np.array(color_map)
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _vertex_array_(
        cls,
        _u_color_map_: TextureStorage,
        _indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            custom_macros=[],
            texture_storages = [
                _u_color_map_
            ],
            uniform_blocks=[],
            indexed_attributes=_indexed_attributes_buffer_
        )

    def render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        pixel_width = self._pixelated_width_.value * ConfigSingleton().pixel_per_unit
        texture_size = (
            int(np.ceil(texture.width / pixel_width)),
            int(np.ceil(texture.height / pixel_width))
        )
        with ColorFramebufferBatch(size=texture_size) as batch:
            batch.color_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._color_map_ = texture
            self._vertex_array_.render(
                #shader_filename="copy",
                #custom_macros=[],
                #texture_storages = [
                #    self._u_color_map_
                #],
                #uniform_blocks=[],
                framebuffer=batch.framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
            self._color_map_ = batch.color_texture
            self._vertex_array_.render(
                #shader_filename="copy",
                #custom_macros=[],
                #texture_storages = [
                #    self._u_color_map_
                #],
                #uniform_blocks=[],
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
