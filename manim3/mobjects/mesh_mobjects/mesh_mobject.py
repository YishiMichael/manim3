from __future__ import annotations


from typing import Self

import moderngl

from ...animatables.animatable.animatable import AnimatableActions
from ...animatables.arrays.animatable_color import AnimatableColor
from ...animatables.arrays.animatable_float import AnimatableFloat
from ...animatables.mesh import Mesh
from ...animatables.model import ModelActions
from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8,
    NP_x3i4
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.attributes_buffer import AttributesBuffer
from ...rendering.buffers.texture_buffer import TextureBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ...utils.path_utils import PathUtils
from ..mobject import Mobject


class MeshMobject(Mobject):
    __slots__ = ()

    def __init__(
        self: Self,
        mesh: Mesh | None = None
    ) -> None:
        super().__init__()
        if mesh is not None:
            self._mesh_ = mesh

    @AnimatableActions.interpolate.register_descriptor()
    @Lazy.volatile()
    @staticmethod
    def _mesh_() -> Mesh:
        return Mesh()

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableColor)
    @Lazy.volatile()
    @staticmethod
    def _color_() -> AnimatableColor:
        return AnimatableColor(Toplevel.config.default_color)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _opacity_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.default_opacity)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _weight_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.default_weight)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _ambient_strength_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.mesh_ambient_strength)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _specular_strength_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.mesh_specular_strength)

    @AnimatableActions.interpolate.register_descriptor()
    @ModelActions.set.register_descriptor(converter=AnimatableFloat)
    @Lazy.volatile()
    @staticmethod
    def _shininess_() -> AnimatableFloat:
        return AnimatableFloat(Toplevel.config.mesh_shininess)

    @Lazy.variable(plural=True)
    @staticmethod
    def _color_maps_() -> tuple[moderngl.Texture, ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _local_sample_positions_(
        mesh__positions: NP_x3f8,
        mesh__faces: NP_x3i4
    ) -> NP_x3f8:
        return mesh__positions[mesh__faces.flatten()]

    @Lazy.property()
    @staticmethod
    def _material_uniform_block_buffer_(
        color__array: NP_3f8,
        opacity__array: NP_f8,
        weight__array: NP_f8,
        ambient_strength__array: NP_f8,
        specular_strength__array: NP_f8,
        shininess__array: NP_f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_material",
            field_declarations=(
                "vec3 u_color",
                "float u_opacity",
                "float u_weight",
                "float u_ambient_strength",
                "float u_specular_strength",
                "float u_shininess"
            ),
            data_dict={
                "u_color": color__array,
                "u_opacity": opacity__array,
                "u_weight": weight__array,
                "u_ambient_strength": ambient_strength__array,
                "u_specular_strength": specular_strength__array,
                "u_shininess": shininess__array
            }
        )

    @Lazy.property()
    @staticmethod
    def _color_maps_texture_buffer_(
        color_maps: tuple[moderngl.Texture, ...]
    ) -> TextureBuffer:
        return TextureBuffer(
            name="t_color_maps",
            textures=color_maps,
            array_lens={
                "NUM_T_COLOR_MAPS": len(color_maps)
            }
        )

    @Lazy.property()
    @staticmethod
    def _mesh_vertex_array_(
        color_maps_texture_buffer: TextureBuffer,
        camera__camera_uniform_block_buffer: UniformBlockBuffer,
        lighting__lighting_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        material_uniform_block_buffer: UniformBlockBuffer,
        mesh__attributes_buffer: AttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_path=PathUtils.shaders_dir.joinpath("mesh.glsl"),
            texture_buffers=(
                color_maps_texture_buffer,
            ),
            uniform_block_buffers=(
                camera__camera_uniform_block_buffer,
                lighting__lighting_uniform_block_buffer,
                model_uniform_block_buffer,
                material_uniform_block_buffer
            ),
            attributes_buffer=mesh__attributes_buffer
        )

    def _render(
        self: Self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._mesh_vertex_array_.render(target_framebuffer)
