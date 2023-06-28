import moderngl
import numpy as np

from ..custom_typing import (
    NP_3f8,
    NP_f8
)
from ..lazy.lazy import Lazy
from ..passes.render_pass import RenderPass
from ..rendering.context import (
    Context,
    ContextState
)
from ..rendering.framebuffer import (
    ColorFramebuffer,
    Framebuffer,
    OITFramebuffer
)
from ..rendering.gl_buffer import TextureIdBuffer
from ..rendering.mgl_enums import ContextFlag
from ..rendering.texture import TextureFactory
from ..rendering.vertex_array import VertexArray
from .mobject import (
    Mobject,
    StyleMeta
)
from .renderable_mobject import RenderableMobject


class FrameMobject(Mobject):
    __slots__ = ()

    @StyleMeta.register()
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.zeros((3,))

    @StyleMeta.register()
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.zeros(())

    @Lazy.variable_collection
    @classmethod
    def _render_passes_(cls) -> list[RenderPass]:
        return []

    @Lazy.property
    @classmethod
    def _copy_vertex_array_(cls) -> VertexArray:
        return VertexArray(
            shader_filename="copy",
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_color_map"
                )
            ]
        )

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

    def _render_scene_content(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        target_framebuffer.framebuffer.clear(
            color=tuple(np.append(self._color_, self._opacity_))
        )
        with TextureFactory.texture(dtype="f2") as accum_texture, \
                TextureFactory.texture(components=1, dtype="f2") as revealage_texture:
            oit_framebuffer = OITFramebuffer(
                accum_texture=accum_texture,
                revealage_texture=revealage_texture
            )
            oit_framebuffer.framebuffer.clear()
            for mobject in self.iter_descendants():
                if isinstance(mobject, RenderableMobject):
                    mobject._render(oit_framebuffer)

            self._oit_compose_vertex_array_.render(
                framebuffer=target_framebuffer,
                texture_array_dict={
                    "t_accum_map": np.array(accum_texture),
                    "t_revealage_map": np.array(revealage_texture)
                },
                context_state=ContextState(
                    flags=(ContextFlag.BLEND,)
                )
            )

    def _render_scene(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        render_passes = self._render_passes_
        if not render_passes:
            self._render_scene_content(target_framebuffer)
            return

        with TextureFactory.texture() as texture_0, TextureFactory.texture() as texture_1:
            framebuffers = (
                ColorFramebuffer(
                    color_texture=texture_0
                ),
                ColorFramebuffer(
                    color_texture=texture_1
                )
            )
            target_id = 0
            self._render_scene_content(framebuffers[0])
            for render_pass in render_passes[:-1]:
                target_id = 1 - target_id
                render_pass._render(
                    texture=framebuffers[1 - target_id].color_texture,
                    target_framebuffer=framebuffers[target_id]
                )
            render_passes[-1]._render(
                texture=framebuffers[target_id].color_texture,
                target_framebuffer=target_framebuffer
            )

    def _render_to_window(
        self,
        color_texture: moderngl.Texture
    ) -> None:
        window = Context.window
        if window.is_closing:
            raise KeyboardInterrupt
        window.clear()
        self._copy_vertex_array_.render(
            framebuffer=Framebuffer(
                framebuffer=Context.window_framebuffer,
                default_context_state=ContextState(
                    flags=()
                )
            ),
            texture_array_dict={
                "t_color_map": np.array(color_texture)
            }
        )
        window.swap_buffers()
