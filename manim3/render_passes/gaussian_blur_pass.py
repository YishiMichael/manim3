__all__ = ["GaussianBlurPass"]


import moderngl
import numpy as np

from ..config import Config
from ..custom_typing import (
    FloatsT,
    Real
)
from ..render_passes.render_pass import RenderPass
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)
from ..utils.render_procedure import (
    UniformBlockBuffer,
    RenderProcedure,
    TextureStorage
)


class GaussianBlurPass(RenderPass):
    def __init__(self, sigma_width: Real = 0.1):
        super().__init__()
        self._sigma_width_ = sigma_width

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property_writable
    @staticmethod
    def _sigma_width_() -> Real:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _convolution_core_(sigma_width: Real) -> FloatsT:
        sigma = sigma_width * Config.pixel_per_unit
        n = int(np.ceil(3.0 * sigma))
        convolution_core = np.exp(-np.arange(n + 1) ** 2 / (2.0 * sigma ** 2))
        return convolution_core / (2.0 * convolution_core.sum() - convolution_core[0])

    @lazy_property
    @staticmethod
    def _ub_gaussian_blur_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_gaussian_blur", [
            "vec2 u_uv_offset",
            "float u_convolution_core[CONVOLUTION_CORE_SIZE]"
        ])

    @lazy_property
    @staticmethod
    def _ub_gaussian_blur_(
        ub_gaussian_blur_o: UniformBlockBuffer,
        convolution_core: FloatsT
    ) -> UniformBlockBuffer:
        return ub_gaussian_blur_o.write({
            "u_uv_offset": 1.0 / np.array(Config.pixel_size),
            "u_convolution_core": convolution_core
        })

    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        with RenderProcedure.texture() as intermediate_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[intermediate_texture],
                    depth_attachment=None
                ) as intermediate_framebuffer:
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("gaussian_blur"),
                custom_macros=[
                    "#define blur_subroutine horizontal_dilate"
                ],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(texture)
                    )
                ],
                uniform_blocks=[
                    self._ub_gaussian_blur_
                ],
                framebuffer=intermediate_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("gaussian_blur"),
                custom_macros=[
                    "#define blur_subroutine vertical_dilate"
                ],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(intermediate_texture)
                    )
                ],
                uniform_blocks=[
                    self._ub_gaussian_blur_
                ],
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
