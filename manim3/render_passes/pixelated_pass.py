__all__ = ["PixelatedPass"]


import moderngl
import numpy as np

from ..constants import PIXEL_PER_UNIT
from ..custom_typing import Real
from ..render_passes.render_pass import RenderPass
from ..utils.lazy import lazy_property
from ..utils.render_procedure import (
    RenderProcedure,
    TextureStorage
)


class PixelatedPass(RenderPass):
    def __init__(self, pixel_width: Real = 0.075):
        self._pixel_width: Real = pixel_width

    @lazy_property
    @staticmethod
    def _u_color_map_o_() -> TextureStorage:
        return TextureStorage("sampler2D u_color_map")

    def _render(self, texture: moderngl.Texture, target_framebuffer: moderngl.Framebuffer) -> None:
        texture_size = (
            int(np.ceil(texture.width / (self._pixel_width * PIXEL_PER_UNIT))),
            int(np.ceil(texture.height / (self._pixel_width * PIXEL_PER_UNIT)))
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
