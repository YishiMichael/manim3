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
from ..utils.lazy import lazy_property


class PixelatedPass(RenderPass):
    def __init__(self, pixelated_width: Real = 0.1):
        super().__init__()
        self._pixelated_width: Real = pixelated_width

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        pixel_width = self._pixelated_width * ConfigSingleton().pixel_per_unit
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
