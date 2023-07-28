import moderngl
import numpy as np

from ..constants.custom_typing import NP_3f8
from ..lazy.lazy import Lazy
from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.texture_buffer import TextureBuffer
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from .mobject.mobject import Mobject
from .renderable_mobject import RenderableMobject


class SceneRootMobject(Mobject):
    __slots__ = ("_background_color",)

    def __init__(
        self,
        background_color: NP_3f8
    ) -> None:
        super().__init__()
        self._oit_framebuffer_ = OITFramebuffer()
        self._background_color: NP_3f8 = background_color

    @Lazy.variable
    @classmethod
    def _oit_framebuffer_(cls) -> OITFramebuffer:
        return NotImplemented

    @Lazy.property
    @classmethod
    def _oit_compose_vertex_array_(
        cls,
        oit_framebuffer__accum_texture: moderngl.Texture,
        oit_framebuffer__revealage_texture: moderngl.Texture
    ) -> VertexArray:
        return VertexArray(
            shader_filename="oit_compose",
            texture_buffers=[
                TextureBuffer(
                    field="sampler2D t_accum_map",
                    texture_array=np.array(oit_framebuffer__accum_texture, dtype=moderngl.Texture)
                ),
                TextureBuffer(
                    field="sampler2D t_revealage_map",
                    texture_array=np.array(oit_framebuffer__revealage_texture, dtype=moderngl.Texture)
                )
            ],
            indexed_attributes_buffer=IndexedAttributesBuffer(
                attributes_buffer=AttributesBuffer(
                    fields=[
                        "vec3 in_position",
                        "vec2 in_uv"
                    ],
                    num_vertex=4,
                    data={
                        "in_position": np.array((
                            (-1.0, -1.0, 0.0),
                            (1.0, -1.0, 0.0),
                            (1.0, 1.0, 0.0),
                            (-1.0, 1.0, 0.0)
                        )),
                        "in_uv": np.array((
                            (0.0, 0.0),
                            (1.0, 0.0),
                            (1.0, 1.0),
                            (0.0, 1.0)
                        ))
                    }
                ),
                mode=PrimitiveMode.TRIANGLE_FAN
            )
        )

    def _render_scene(
        self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        target_framebuffer._framebuffer_.clear(
            color=tuple(self._background_color)
        )

        oit_framebuffer = self._oit_framebuffer_
        oit_framebuffer._framebuffer_.clear()
        for mobject in self.iter_descendants():
            if isinstance(mobject, RenderableMobject):
                mobject._render(oit_framebuffer)

        self._oit_compose_vertex_array_.render(target_framebuffer)
