__all__ = ["GaussianBlurPass"]


import moderngl
import numpy as np

from ..custom_typing import (
    FloatsT,
    Real
)
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.render_procedure import (
    UniformBlockBuffer,
    RenderProcedure,
    TextureStorage
)
from ..utils.lazy import (
    LazyData,
    lazy_basedata,
    lazy_property
)


#class GaussianBlurPass(RenderPass):
#    def __init__(self, sigma_width: Real = 0.1):
#        self._sigma_width: Real = sigma_width
#
#    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
#        instance = GaussianBlurPassSingleton()
#        instance._sigma_width_ = self._sigma_width
#        instance.render(texture, target_framebuffer)


class GaussianBlurPass(RenderPass):
    def __new__(cls, sigma_width: Real | None = None):
        instance = super().__new__(cls)
        if sigma_width is not None:
            instance._sigma_width_ = LazyData(sigma_width)
        return instance

    @lazy_basedata
    @staticmethod
    def _sigma_width_() -> Real:
        return 0.1

    @lazy_property
    @staticmethod
    def _convolution_core_(sigma_width: Real) -> FloatsT:
        sigma = sigma_width * ConfigSingleton().pixel_per_unit
        n = int(np.ceil(3.0 * sigma))
        convolution_core = np.exp(-np.arange(n + 1) ** 2 / (2.0 * sigma ** 2))
        return convolution_core / (2.0 * convolution_core.sum() - convolution_core[0])

    @lazy_basedata
    @staticmethod
    def _color_map_() -> moderngl.Texture:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _u_color_map_(color_map: moderngl.Texture) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_color_map",
            texture_array=np.array(color_map)
        )

    #@lazy_property
    #@staticmethod
    #def _ub_gaussian_blur_o_() -> UniformBlockBuffer:
    #    return UniformBlockBuffer("ub_gaussian_blur", [
    #        "vec2 u_uv_offset",
    #        "float u_convolution_core[CONVOLUTION_CORE_SIZE]"
    #    ])

    @lazy_property
    @staticmethod
    def _ub_gaussian_blur_(
        #ub_gaussian_blur_o: UniformBlockBuffer,
        convolution_core: FloatsT
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_gaussian_blur",
            fields=[
                "vec2 u_uv_offset",
                "float u_convolution_core[CONVOLUTION_CORE_SIZE]"
            ],
            data={
                "u_uv_offset": 1.0 / np.array(ConfigSingleton().pixel_size),
                "u_convolution_core": convolution_core
            }
        )

    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        with RenderProcedure.texture() as intermediate_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[intermediate_texture],
                    depth_attachment=None
                ) as intermediate_framebuffer:
            self._color_map_ = LazyData(texture)
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("gaussian_blur"),
                custom_macros=[
                    "#define blur_subroutine horizontal_dilate"
                ],
                texture_storages=[
                    self._u_color_map_
                ],
                uniform_blocks=[
                    self._ub_gaussian_blur_
                ],
                framebuffer=intermediate_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
            self._color_map_ = LazyData(intermediate_texture)
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("gaussian_blur"),
                custom_macros=[
                    "#define blur_subroutine vertical_dilate"
                ],
                texture_storages=[
                    self._u_color_map_
                ],
                uniform_blocks=[
                    self._ub_gaussian_blur_
                ],
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
