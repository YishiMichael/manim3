import moderngl
import numpy as np

from ..constants.custom_typing import NP_3f8
from ..lazy.lazy import Lazy
from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.texture_id_buffer import TextureIdBuffer
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from .mobject import Mobject
from .renderable_mobject import RenderableMobject


class SceneRootMobject(Mobject):
    __slots__ = (
        "_background_color",
        "_oit_framebuffer"
    )

    def __init__(
        self,
        background_color: NP_3f8
    ) -> None:
        super().__init__()
        self._background_color: NP_3f8 = background_color
        self._oit_framebuffer: OITFramebuffer = OITFramebuffer()

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
        target_framebuffer.framebuffer.clear(
            color=tuple(self._background_color)
        )

        oit_framebuffer = self._oit_framebuffer
        oit_framebuffer.framebuffer.clear()
        for mobject in self.iter_descendants():
            if isinstance(mobject, RenderableMobject):
                mobject._render(oit_framebuffer)

        self._oit_compose_vertex_array_.render(
            framebuffer=target_framebuffer,
            texture_array_dict={
                "t_accum_map": np.array(oit_framebuffer.accum_texture, dtype=moderngl.Texture),
                "t_revealage_map": np.array(oit_framebuffer.revealage_texture, dtype=moderngl.Texture)
            }
        )
