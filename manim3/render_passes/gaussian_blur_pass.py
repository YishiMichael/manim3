__all__ = ["GaussianBlurPass"]


import moderngl
import numpy as np

from ..custom_typing import (
    FloatsT,
    Real
)
from ..render_passes.render_pass import RenderPass
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)
from ..utils.renderable import (
    AttributesBuffer,
    #ContextState,
    #Framebuffer,
    IndexBuffer,
    #IntermediateTextures,
    UniformBlockBuffer,
    RenderProcedure,
    #RenderStep,
    TextureStorage
)


class GaussianBlurPass(RenderPass):
    def __init__(self, sigma: Real = 1.0):
        self.sigma: Real = sigma

    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        GaussianBlurPassProcedure().set(
            sigma=self.sigma
        ).render(texture, target_framebuffer)


class GaussianBlurPassProcedure(RenderProcedure):
    @lazy_property
    @staticmethod
    def _intermediate_texture_() -> moderngl.Texture:
        return RenderProcedure.construct_texture()

    @lazy_property
    @staticmethod
    def _intermediate_framebuffer_(
        intermediate_texture: moderngl.Texture
    ) -> moderngl.Framebuffer:
        return RenderProcedure.construct_framebuffer(
            color_attachments=[intermediate_texture],
            depth_attachment=None
        )

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    @lazy_property
    @staticmethod
    def _attributes_() -> AttributesBuffer:
        return AttributesBuffer([
            "vec3 in_position",
            "vec2 in_uv"
        ]).write({
            "in_position": np.array([
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]),
            "in_uv": np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ])
        })

    @lazy_property
    @staticmethod
    def _index_buffer_() -> IndexBuffer:
        return IndexBuffer().write(np.array((
            0, 1, 2, 3
        )))

    @lazy_property_writable
    @staticmethod
    def _sigma_() -> Real:
        return 1.0

    @lazy_property
    @staticmethod
    def _convolution_core_(sigma: Real) -> FloatsT:
        n = int(np.ceil(3.0 * sigma))
        convolution_core = np.exp(-np.arange(-n, n + 1) ** 2 / (2.0 * sigma ** 2))
        return convolution_core[n:] / convolution_core.sum()

    @lazy_property
    @staticmethod
    def _ub_convolution_core_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer("ub_convolution_core", [
            "float u_convolution_core[CONVOLUTION_CORE_SIZE]"
        ])

    @lazy_property
    @staticmethod
    def _ub_convolution_core_(
        ub_convolution_core_o: UniformBlockBuffer,
        convolution_core: FloatsT
    ) -> UniformBlockBuffer:
        ub_convolution_core_o.write({
            "u_convolution_core": convolution_core
        })
        return ub_convolution_core_o

    def set(
        self,
        *,
        sigma: Real
    ):
        self._sigma_ = sigma
        return self

    def render(
        self,
        texture: moderngl.Texture,
        target_framebuffer: moderngl.Framebuffer
    ) -> None:
        self.render_by_step(self.render_step(
            shader_str=self.read_shader("gaussian_blur"),
            custom_macros=[
                "#define blur_subroutine horizontal_dilate"
            ],
            texture_storages=[
                self._u_color_map_o_.write(
                    np.array(texture)
                )
            ],
            uniform_blocks=[
                self._ub_convolution_core_
            ],
            attributes=self._attributes_,
            index_buffer=self._index_buffer_,
            framebuffer=self._intermediate_framebuffer_,
            enable_only=moderngl.NOTHING,
            context_state=self.context_state(),
            mode=moderngl.TRIANGLE_FAN
        ), self.render_step(
            shader_str=self.read_shader("gaussian_blur"),
            custom_macros=[
                "#define blur_subroutine vertical_dilate"
            ],
            texture_storages=[
                self._u_color_map_o_.write(
                    np.array(self._intermediate_texture_)
                )
            ],
            uniform_blocks=[
                self._ub_convolution_core_
            ],
            attributes=self._attributes_,
            index_buffer=self._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=moderngl.NOTHING,
            context_state=self.context_state(),
            mode=moderngl.TRIANGLE_FAN
        ))
