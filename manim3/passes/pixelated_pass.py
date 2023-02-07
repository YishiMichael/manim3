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


class PixelatedPass(RenderPass):
    __slots__ = ()

    def __new__(cls, pixelated_width: Real | None = None):
        instance = super().__new__(cls)
        if pixelated_width is not None:
            instance._pixelated_width_ = LazyData(pixelated_width)
        return instance

    @lazy_basedata
    @staticmethod
    def _pixelated_width_() -> Real:
        return 0.1

    @lazy_basedata
    @staticmethod
    def _color_map_() -> moderngl.Texture:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _u_color_map_(
        color_map: moderngl.Texture
    ) -> TextureStorage:
        return TextureStorage(
            field="sampler2D u_color_map",
            texture_array=np.array(color_map)
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
            self._color_map_ = LazyData(texture)
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_
                ],
                uniform_blocks=[],
                framebuffer=intermediate_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
            self._color_map_ = LazyData(intermediate_texture)
            RenderProcedure.fullscreen_render_step(
                shader_str=RenderProcedure.read_shader("copy"),
                custom_macros=[],
                texture_storages=[
                    self._u_color_map_
                ],
                uniform_blocks=[],
                framebuffer=target_framebuffer,
                context_state=RenderProcedure.context_state(
                    enable_only=moderngl.NOTHING
                )
            )
