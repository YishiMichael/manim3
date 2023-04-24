import moderngl
import numpy as np

from ..lazy.interface import Lazy
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.framebuffer import ColorFramebuffer
from ..rendering.gl_buffer import TextureIDBuffer
from ..rendering.texture import TextureFactory
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

    @Lazy.variable_external
    @classmethod
    def _pixelated_width_(cls) -> float:
        return 0.1

    @Lazy.property
    @classmethod
    def _pixelated_va_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            texture_id_buffers=[
                TextureIDBuffer(
                    field="sampler2D t_color_map"
                )
            ]
        )

    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        pixel_width = self._pixelated_width_.value * ConfigSingleton().size.pixel_per_unit
        texture_size = (
            int(np.ceil(texture.width / pixel_width)),
            int(np.ceil(texture.height / pixel_width))
        )
        with TextureFactory.texture(size=texture_size) as color_texture:
            color_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)  # TODO: typing
            self._pixelated_va_.render(
                framebuffer=ColorFramebuffer(
                    color_texture=color_texture
                ),
                texture_array_dict={
                    "t_color_map": np.array(texture)
                }
            )
            self._pixelated_va_.render(
                framebuffer=target_framebuffer,
                texture_array_dict={
                    "t_color_map": np.array(color_texture)
                }
            )
