__all__ = ["PixelatedPass"]


import moderngl
import numpy as np

from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.context import ContextState
from ..rendering.gl_buffer import TextureIDBuffer
from ..rendering.temporary_resource import ColorFramebufferBatch
from ..rendering.vertex_array import VertexArray


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

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _u_color_map_(cls) -> TextureIDBuffer:
        return TextureIDBuffer(
            field="sampler2D u_color_map"
        )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _vertex_array_(
        cls,
        _u_color_map_: TextureIDBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            #custom_macros=[],
            texture_id_buffers=[
                _u_color_map_
            ]
            #uniform_block_buffers=[]
        )

    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        pixel_width = self._pixelated_width_.value * ConfigSingleton().size.pixel_per_unit
        texture_size = (
            int(np.ceil(texture.width / pixel_width)),
            int(np.ceil(texture.height / pixel_width))
        )
        with ColorFramebufferBatch(size=texture_size) as batch:
            batch.color_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self._vertex_array_.render(
                texture_array_dict={
                    "u_color_map": np.array(texture)
                },
                framebuffer=batch.framebuffer,
                context_state=ContextState(
                    flags=()
                )
            )
            self._vertex_array_.render(
                texture_array_dict={
                    "u_color_map": np.array(batch.color_texture)
                },
                framebuffer=target_framebuffer,
                context_state=ContextState(
                    flags=()
                )
            )
