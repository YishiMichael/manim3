import numpy as np

from ..custom_typing import NP_3f8
from ..lazy.lazy import Lazy
from ..rendering.framebuffer import (
    ColorFramebuffer,
    OITFramebuffer
)
from ..rendering.gl_buffer import TextureIdBuffer
from ..rendering.texture import TextureFactory
from ..rendering.vertex_array import VertexArray
from .mobject_style_meta import MobjectStyleMeta
from .renderable_mobject import RenderableMobject


class SceneRootMobject(RenderableMobject):
    __slots__ = ()

    @MobjectStyleMeta.register()
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.zeros((3,))

    @Lazy.property
    @classmethod
    def _oit_compose_vertex_array_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_accum_map"
                ),
                TextureIdBuffer(
                    field="sampler2D t_revealage_map"
                )
            ]
        )

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        for mobject in self.iter_real_descendants():
            if isinstance(mobject, RenderableMobject):
                mobject._render(target_framebuffer)

    def _render_scene(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        target_framebuffer.framebuffer.clear(
            color=tuple(self._color_)
        )
        with TextureFactory.texture(components=4, dtype="f2") as accum_texture, \
                TextureFactory.texture(components=1, dtype="f2") as revealage_texture:
            oit_framebuffer = OITFramebuffer(
                accum_texture=accum_texture,
                revealage_texture=revealage_texture
            )
            oit_framebuffer.framebuffer.clear()
            self._render(oit_framebuffer)
            self._oit_compose_vertex_array_.render(
                framebuffer=target_framebuffer,
                texture_array_dict={
                    "t_accum_map": np.array(accum_texture),
                    "t_revealage_map": np.array(revealage_texture)
                }
            )
