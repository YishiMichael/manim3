from __future__ import annotations


from typing import Self

import moderngl
import numpy as np

from ..animatables.arrays.animatable_color import AnimatableColor
from ..animatables.arrays.animatable_float import AnimatableFloat
from ..lazy.lazy import Lazy
from ..rendering.buffers.attributes_buffer import AttributesBuffer
from ..rendering.buffers.texture_buffer import TextureBuffer
from ..rendering.framebuffers.color_framebuffer import ColorFramebuffer
from ..rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ..rendering.mgl_enums import PrimitiveMode
from ..rendering.vertex_array import VertexArray
from ..toplevel.toplevel import Toplevel
from ..utils.path_utils import PathUtils
from .mobject import Mobject


class SceneRootMobject(Mobject):
    __slots__ = ()

    @Lazy.volatile()
    @staticmethod
    def _background_color_() -> AnimatableColor:
        return AnimatableColor(Toplevel.config.background_color)

    @Lazy.volatile()
    @staticmethod
    def _background_opacity_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.background_opacity)

    @Lazy.property()
    @staticmethod
    def _oit_framebuffer_() -> OITFramebuffer:
        return OITFramebuffer()

    @Lazy.property()
    @staticmethod
    def _oit_compose_vertex_array_(
        oit_framebuffer__accum_texture: moderngl.Texture,
        oit_framebuffer__revealage_texture: moderngl.Texture
    ) -> VertexArray:
        return VertexArray(
            shader_path=PathUtils.shaders_dir.joinpath("oit_compose.glsl"),
            texture_buffers=(
                TextureBuffer(
                    name="t_accum_map",
                    textures=oit_framebuffer__accum_texture
                ),
                TextureBuffer(
                    name="t_revealage_map",
                    textures=oit_framebuffer__revealage_texture
                )
            ),
            attributes_buffer=AttributesBuffer(
                field_declarations=(
                    "vec3 in_position",
                    "vec2 in_uv"
                ),
                data_dict={
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
                },
                primitive_mode=PrimitiveMode.TRIANGLE_FAN,
                num_vertices=4
            )
        )

    def _get_vertex_array(
        self: Self
    ) -> VertexArray | None:
        return self._oit_compose_vertex_array_

    def _render_scene(
        self: Self,
        target_framebuffer: ColorFramebuffer
    ) -> None:
        red, green, blue = self._background_color_._array_
        alpha = self._background_opacity_._array_
        target_framebuffer._framebuffer_.clear(
            red=float(red), green=float(green), blue=float(blue), alpha=float(alpha)
        )

        oit_framebuffer = self._oit_framebuffer_
        oit_framebuffer._framebuffer_.clear()
        for child in self.iter_children():
            for mobject in child.iter_descendants():
                mobject._render(oit_framebuffer)

        self._render(target_framebuffer)
        #self._oit_compose_vertex_array_.render(target_framebuffer)
