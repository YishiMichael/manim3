__all__ = ["MeshMobject"]


import moderngl
import numpy as np

from ..custom_typing import (
    ColorType,
    Vec3T,
    Vec3sT
)
from ..geometries.geometry import Geometry
from ..lazy.core import LazyWrapper
from ..lazy.interface import Lazy
from ..mobjects.mobject import Mobject
from ..rendering.framebuffer import (
    TransparentFramebuffer,
    OpaqueFramebuffer
)
from ..rendering.gl_buffer import (
    TextureIDBuffer,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.color import ColorUtils


class MeshMobject(Mobject):
    __slots__ = ()

    @Lazy.variable
    @classmethod
    def _geometry_(cls) -> Geometry:
        return Geometry()

    @Lazy.variable_external
    @classmethod
    def _color_map_(cls) -> moderngl.Texture | None:
        return None

    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @Lazy.variable_external
    @classmethod
    def _ambient_strength_(cls) -> float:
        return 1.0

    @Lazy.variable_external
    @classmethod
    def _specular_strength_(cls) -> float:
        return 0.5

    @Lazy.variable_external
    @classmethod
    def _shininess_(cls) -> float:
        return 32.0

    @Lazy.variable_shared
    @classmethod
    def _apply_phong_lighting_(cls) -> bool:
        return True

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(
        cls,
        _geometry_: Geometry
    ) -> Vec3sT:
        return _geometry_._geometry_data_.value.position

    @Lazy.property
    @classmethod
    def _material_uniform_block_buffer_(
        cls,
        color: Vec3T,
        opacity: float,
        ambient_strength: float,
        specular_strength: float,
        shininess: float
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_material",
            fields=[
                "vec4 u_color",
                "float u_ambient_strength",
                "float u_specular_strength",
                "float u_shininess"
            ],
            data={
                "u_color": np.append(color, opacity),
                "u_ambient_strength": np.array(ambient_strength),
                "u_specular_strength": np.array(specular_strength),
                "u_shininess": np.array(shininess)
            }
        )

    @Lazy.property
    @classmethod
    def _mesh_vertex_array_(
        cls,
        is_transparent: bool,
        apply_phong_lighting: bool,
        color_map: moderngl.Texture | None,
        _scene_state__camera__camera_uniform_block_buffer_: UniformBlockBuffer,
        _model_uniform_block_buffer_: UniformBlockBuffer,
        _scene_state__lights_uniform_block_buffer_: UniformBlockBuffer,
        _material_uniform_block_buffer_: UniformBlockBuffer,
        _geometry__indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        custom_macros: list[str] = []
        if is_transparent:
            custom_macros.append("#define IS_TRANSPARENT")
        phong_lighting_subroutine = "enable_phong_lighting" if apply_phong_lighting else "disable_phong_lighting"
        custom_macros.append(f"#define phong_lighting_subroutine {phong_lighting_subroutine}")
        return VertexArray(
            shader_filename="mesh",
            custom_macros=custom_macros,
            texture_id_buffers=[
                TextureIDBuffer(
                    field="sampler2D t_color_maps[NUM_COLOR_MAPS]",
                    array_lens={
                        "NUM_COLOR_MAPS": int(color_map is not None)
                    }
                )
            ],
            uniform_block_buffers=[
                _scene_state__camera__camera_uniform_block_buffer_,
                _model_uniform_block_buffer_,
                _scene_state__lights_uniform_block_buffer_,
                _material_uniform_block_buffer_
            ],
            indexed_attributes_buffer=_geometry__indexed_attributes_buffer_
        )

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        textures: list[moderngl.Texture] = []
        if (color_map := self._color_map_.value) is not None:
            textures.append(color_map)
        self._mesh_vertex_array_.render(
            framebuffer=target_framebuffer,
            texture_array_dict={
                "t_color_maps": np.array(textures, dtype=moderngl.Texture)
            }
        )

    def get_geometry(self) -> Geometry:
        return self._geometry_

    def set_geometry(
        self,
        geometry: Geometry
    ):
        self._geometry_ = geometry
        return self

    def set_style(
        self,
        *,
        color: ColorType | None = None,
        opacity: float | None = None,
        is_transparent: bool | None = None,
        ambient_strength: float | None = None,
        specular_strength: float | None = None,
        shininess: float | None = None,
        apply_phong_lighting: bool | None = None,
        broadcast: bool = True
    ):
        color_component, opacity_component = ColorUtils.normalize_color_input(color, opacity)
        color_value = LazyWrapper(color_component) if color_component is not None else None
        opacity_value = LazyWrapper(opacity_component) if opacity_component is not None else None
        is_transparent_value = is_transparent if is_transparent is not None else \
            True if opacity_component is not None else None
        ambient_strength_value = LazyWrapper(ambient_strength) if ambient_strength is not None else None
        specular_strength_value = LazyWrapper(specular_strength) if specular_strength is not None else None
        shininess_value = LazyWrapper(shininess) if shininess is not None else None
        apply_phong_lighting_value = apply_phong_lighting if apply_phong_lighting is not None else \
            True if any(param is not None for param in (
                ambient_strength,
                specular_strength,
                shininess
            )) else None
        for mobject in self.iter_descendants_by_type(mobject_type=MeshMobject, broadcast=broadcast):
            if color_value is not None:
                mobject._color_ = color_value
            if opacity_value is not None:
                mobject._opacity_ = opacity_value
            if is_transparent_value is not None:
                mobject._is_transparent_ = is_transparent_value
            if ambient_strength_value is not None:
                mobject._ambient_strength_ = ambient_strength_value
            if specular_strength_value is not None:
                mobject._specular_strength_ = specular_strength_value
            if shininess_value is not None:
                mobject._shininess_ = shininess_value
            if apply_phong_lighting_value is not None:
                mobject._apply_phong_lighting_ = apply_phong_lighting_value
        return self
