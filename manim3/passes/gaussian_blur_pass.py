import moderngl
import numpy as np

from ..config import ConfigSingleton
from ..custom_typing import NP_xf8
from ..lazy.lazy import Lazy
from ..passes.render_pass import RenderPass
from ..rendering.framebuffer import ColorFramebuffer
from ..rendering.gl_buffer import (
    UniformBlockBuffer,
    TextureIdBuffer
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

    @Lazy.variable_external
    @classmethod
    def _sigma_width_(cls) -> float:
        return 0.1

    @Lazy.property_external
    @classmethod
    def _convolution_core_(
        cls,
        sigma_width: float
    ) -> NP_xf8:
        sigma = sigma_width * ConfigSingleton().size.pixel_per_unit
        n = int(np.ceil(3.0 * sigma))
        convolution_core = np.exp(-np.arange(n + 1) ** 2 / (2.0 * sigma ** 2))
        return convolution_core / (2.0 * convolution_core.sum() - convolution_core[0])

    @Lazy.property
    @classmethod
    def _gaussian_blur_uniform_block_buffer_(
        cls,
        convolution_core: NP_xf8
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

    @Lazy.property_collection
    @classmethod
    def _gaussian_blur_vertex_arrays_(
        cls,
        gaussian_blur_uniform_block_buffer: UniformBlockBuffer
    ) -> list[VertexArray]:
        return [
            VertexArray(
                shader_filename="gaussian_blur",
                custom_macros=[
                    f"#define blur_subroutine {blur_subroutine}"
                ],
                texture_id_buffers=[
                    TextureIdBuffer(
                        field="sampler2D t_color_map"
                    )
                ],
                uniform_block_buffers=[
                    gaussian_blur_uniform_block_buffer
                ]
            )
            for blur_subroutine in ("horizontal_dilate", "vertical_dilate")
        ]

    def _render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        with TextureFactory.texture() as color_texture:
            self._gaussian_blur_vertex_arrays_[0].render(
                framebuffer=ColorFramebuffer(
                    color_texture=color_texture
                ),
                texture_array_dict={
                    "t_color_map": np.array(texture)
                }
            )
            self._gaussian_blur_vertex_arrays_[1].render(
                framebuffer=target_framebuffer,
                texture_array_dict={
                    "t_color_map": np.array(color_texture)
                }
            )
