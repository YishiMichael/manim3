__all__ = ["PixelatedPass"]


import moderngl
import numpy as np

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
    VertexArray
)


class PixelatedPass(RenderPass):
    __slots__ = ()

    def __init__(
        self,
        pixelated_width: float | None = None
    ) -> None:
        super().__init__()
        if pixelated_width is not None:
            self._pixelated_width_ = pixelated_width

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _pixelated_width_(cls) -> float:
        return 0.1

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _u_color_map_(cls) -> TextureStorage:
        return TextureStorage()

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _vertex_array_(cls) -> VertexArray:
        return VertexArray()

    def _render(
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
            self._vertex_array_.write(
                shader_filename="copy",
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_.write(
                        field="sampler2D u_color_map",
                        texture_array=np.array(texture)
                    )
                ],
                uniform_blocks=[],
                indexed_attributes=self._indexed_attributes_buffer_
            ).render(
                framebuffer=batch.framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
            self._vertex_array_.write(
                shader_filename="copy",
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_.write(
                        field="sampler2D u_color_map",
                        texture_array=np.array(batch.color_texture)
                    )
                ],
                uniform_blocks=[],
                indexed_attributes=self._indexed_attributes_buffer_
            ).render(
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    enable_only=moderngl.NOTHING
                )
            )
