from __future__ import annotations


from typing import Self

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
from ..utils.path_utils import PathUtils
from .mobject import Mobject


class SceneRootMobject(Mobject):
    __slots__ = ("_background_color",)

    def __init__(
        self: Self,
        background_color: NP_3f8
    ) -> None:
        super().__init__()
        self._oit_framebuffer_ = OITFramebuffer()
        self._background_color: NP_3f8 = background_color

    @Lazy.variable()
    @staticmethod
    def _oit_framebuffer_() -> OITFramebuffer:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _oit_compose_vertex_array_(
        oit_framebuffer__accum_texture: moderngl.Texture,
        oit_framebuffer__revealage_texture: moderngl.Texture
    ) -> VertexArray:
        return VertexArray(
            shader_path=PathUtils.shaders_dir.joinpath("oit_compose.glsl"),
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
        self: Self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        target_framebuffer._framebuffer_.clear(
            color=tuple(self._background_color)
        )

        oit_framebuffer = self._oit_framebuffer_
        oit_framebuffer._framebuffer_.clear()
        for mobject in self.iter_descendants():
            mobject._render(oit_framebuffer)

        self._oit_compose_vertex_array_.render(target_framebuffer)
