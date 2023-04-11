__all__ = ["GaussianBlurPass"]


import moderngl
import numpy as np

from ..custom_typing import FloatsT
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
#from ..rendering.context import ContextState
from ..rendering.framebuffer import ColorFramebuffer
from ..rendering.gl_buffer import (
    UniformBlockBuffer,
    TextureIDBuffer
)
from ..rendering.texture import TextureFactory
from ..rendering.vertex_array import VertexArray


class GaussianBlurPass(RenderPass):
    __slots__ = ()

    def __init__(
        self,
        sigma_width: float | None = None
    ) -> None:
        super().__init__()
        if sigma_width is not None:
            self._sigma_width_ = sigma_width

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _sigma_width_(cls) -> float:
        return 0.1

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _convolution_core_(
        cls,
        sigma_width: float
    ) -> FloatsT:
        sigma = sigma_width * ConfigSingleton().size.pixel_per_unit
        n = int(np.ceil(3.0 * sigma))
        convolution_core = np.exp(-np.arange(n + 1) ** 2 / (2.0 * sigma ** 2))
        return convolution_core / (2.0 * convolution_core.sum() - convolution_core[0])

    #@Lazy.property(LazyMode.OBJECT)
    #@classmethod
    #def _color_map_tid_(cls) -> TextureIDBuffer:
    #    return TextureIDBuffer(
    #        field="sampler2D t_color_map"
    #    )

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _gaussian_blur_uniform_block_buffer_(
        cls,
        convolution_core: FloatsT
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_gaussian_blur",
            fields=[
                "vec2 u_uv_offset",
                "float u_convolution_core[CONVOLUTION_CORE_SIZE]"
            ],
            array_lens={
                "CONVOLUTION_CORE_SIZE": len(convolution_core)
            },
            data={
                "u_uv_offset": 1.0 / np.array(ConfigSingleton().size.pixel_size),
                "u_convolution_core": convolution_core
            }
        )

    @Lazy.property(LazyMode.COLLECTION)
    @classmethod
    def _gaussian_blur_vertex_arrays_(
        cls,
        #_color_map_tid_: TextureIDBuffer,
        _gaussian_blur_uniform_block_buffer_: UniformBlockBuffer
    ) -> list[VertexArray]:
        return [
            VertexArray(
                shader_filename="gaussian_blur",
                custom_macros=[
                    f"#define blur_subroutine {blur_subroutine}"
                ],
                texture_id_buffers=[
                    TextureIDBuffer(
                        field="sampler2D t_color_map"
                    )
                ],
                uniform_block_buffers=[
                    _gaussian_blur_uniform_block_buffer_
                ]
            )
            for blur_subroutine in ("horizontal_dilate", "vertical_dilate")
        ]

    #@Lazy.property(LazyMode.OBJECT)
    #@classmethod
    #def _vertical_va_(
    #    cls,
    #    _color_map_tid_: TextureIDBuffer,
    #    _gaussian_blur_uniform_block_buffer_: UniformBlockBuffer
    #) -> VertexArray:
    #    return VertexArray(
    #        shader_filename="gaussian_blur",
    #        custom_macros=[
    #            "#define blur_subroutine vertical_dilate"
    #        ],
    #        texture_id_buffers=[
    #            _color_map_tid_
    #        ],
    #        uniform_block_buffers=[
    #            _gaussian_blur_uniform_block_buffer_
    #        ]
    #    )

    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        with TextureFactory.texture() as color_texture:
            self._gaussian_blur_vertex_arrays_[0].render(
                texture_array_dict={
                    "t_color_map": np.array(texture)
                },
                framebuffer=ColorFramebuffer(
                    color_texture=color_texture
                )
            )
            self._gaussian_blur_vertex_arrays_[1].render(
                texture_array_dict={
                    "t_color_map": np.array(color_texture)
                },
                framebuffer=target_framebuffer
            )
