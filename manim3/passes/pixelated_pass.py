__all__ = ["PixelatedPass"]


import moderngl
import numpy as np

from ..custom_typing import Real
from ..passes.render_pass import RenderPass
from ..rendering.config import ConfigSingleton
from ..rendering.render_procedure import (
    RenderProcedure,
    TextureStorage
)
from ..utils.lazy import (
    LazyData,
    lazy_basedata,
    lazy_property
)


#class PixelatedPass(RenderPass):
#    def __init__(self, pixelated_width: Real = 0.1):
#        self._pixelated_width: Real = pixelated_width
#
#    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
#        instance = PixelatedPassSingleton()
#        instance._pixelated_width_ = self._pixelated_width
#        instance.render(texture, target_framebuffer)


class PixelatedPass(RenderPass):
    def __new__(cls, pixelated_width: Real | None = None):
        instance = super().__new__(cls)
        if pixelated_width is not None:
            instance._pixelated_width_ = LazyData(pixelated_width)
        return instance

    @lazy_basedata
    @staticmethod
    def _pixelated_width_() -> Real:
        return 0.1

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage(
            "sampler2D u_color_map"
        )

    def render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        pixel_width = self._pixelated_width_ * ConfigSingleton().pixel_per_unit
        texture_size = (
            int(np.ceil(texture.width / pixel_width)),
            int(np.ceil(texture.height / pixel_width))
        )
        with RenderProcedure.texture(size=texture_size) as intermediate_texture, \
                RenderProcedure.framebuffer(
                    color_attachments=[intermediate_texture],
                    depth_attachment=None
                ) as intermediate_framebuffer:
            intermediate_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(texture)
                    )
                ],
                uniform_blocks=[],
                framebuffer=intermediate_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_o_.write(
                        np.array(intermediate_texture)
                    )
                ],
                uniform_blocks=[],
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
