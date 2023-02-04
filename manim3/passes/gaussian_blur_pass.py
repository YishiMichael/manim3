__all__ = ["GaussianBlurPass"]


import moderngl
import numpy as np

from ..custom_typing import (
    FloatsT,
    Real
)
from ..passes.render_pass import (
    RenderPass,
    RenderPassSingleton
)
from ..rendering.config import ConfigSingleton
from ..rendering.render_procedure import (
    UniformBlockBuffer,
    RenderProcedure,
    TextureStorage
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)


class GaussianBlurPass(RenderPass):
    def __init__(self, sigma_width: Real = 0.1):
        self._sigma_width: Real = sigma_width

    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        instance = GaussianBlurPassSingleton()
        instance._sigma_width_ = self._sigma_width
        instance.render(texture, target_framebuffer)


class GaussianBlurPassSingleton(RenderPassSingleton):
    @lazy_property_writable
    @staticmethod
    def _sigma_width_() -> Real:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property
    @staticmethod
    def _convolution_core_(sigma_width: Real) -> FloatsT:
        sigma = sigma_width * ConfigSingleton().pixel_per_unit
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
            "u_uv_offset": 1.0 / np.array(ConfigSingleton().pixel_size),
            "u_convolution_core": convolution_core
        })

    def render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
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
