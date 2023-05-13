import moderngl
import numpy as np

from ..custom_typing import (
    Vec3T,
    Vec3sT
)
from ..geometries.geometry import Geometry
from ..lazy.lazy import Lazy
from ..lighting.lighting import Lighting
from ..mobjects.mobject import (
    Mobject,
    MobjectMeta
)
from ..rendering.framebuffer import (
    OpaqueFramebuffer,
    TransparentFramebuffer
)
from ..rendering.gl_buffer import (
    TextureIdBuffer,
    UniformBlockBuffer
)
from ..rendering.vertex_array import (
    IndexedAttributesBuffer,
    VertexArray
)
from ..utils.space import SpaceUtils


class MeshMobject(Mobject):
    __slots__ = ()

    @MobjectMeta.register(
        interpolate_method=NotImplemented
    )
    @Lazy.variable
    @classmethod
    def _geometry_(cls) -> Geometry:
        return Geometry()

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((Mobject._is_transparent_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @MobjectMeta.register(
        interpolate_method=NotImplemented
    )
    @Lazy.variable_external
    @classmethod
    def _color_map_(cls) -> moderngl.Texture | None:
        return None

    @MobjectMeta.register(
        interpolate_method=NotImplemented
    )
    @Lazy.variable_shared
    @classmethod
    def _enable_phong_lighting_(cls) -> bool:
        return True

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((_enable_phong_lighting_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _ambient_strength_(cls) -> float:
        return 1.0

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((_enable_phong_lighting_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _specular_strength_(cls) -> float:
        return 0.5

    @MobjectMeta.register(
        interpolate_method=SpaceUtils.lerp_float,
        related_styles=((_enable_phong_lighting_, True),)
    )
    @Lazy.variable_external
    @classmethod
    def _shininess_(cls) -> float:
        return 32.0

    @Lazy.variable
    @classmethod
    def _lighting_(cls) -> Lighting:  # Keep updated with `Scene._lighting`.
        return Lighting()

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
        enable_phong_lighting: bool,
        color_map: moderngl.Texture | None,
        _camera__camera_uniform_block_buffer_: UniformBlockBuffer,
        _lighting__lighting_uniform_block_buffer_: UniformBlockBuffer,
        _model_uniform_block_buffer_: UniformBlockBuffer,
        _material_uniform_block_buffer_: UniformBlockBuffer,
        _geometry__indexed_attributes_buffer_: IndexedAttributesBuffer
    ) -> VertexArray:
        custom_macros: list[str] = []
        if is_transparent:
            custom_macros.append("#define IS_TRANSPARENT")
        phong_lighting_subroutine = "enable_phong_lighting" if enable_phong_lighting else "disable_phong_lighting"
        custom_macros.append(f"#define phong_lighting_subroutine {phong_lighting_subroutine}")
        return VertexArray(
            shader_filename="mesh",
            custom_macros=custom_macros,
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_color_maps[NUM_T_COLOR_MAPS]",
                    array_lens={
                        "NUM_T_COLOR_MAPS": int(color_map is not None)
                    }
                )
            ],
            uniform_block_buffers=[
                _camera__camera_uniform_block_buffer_,
                _lighting__lighting_uniform_block_buffer_,
                _model_uniform_block_buffer_,
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

    #def get_geometry(self) -> Geometry:
    #    return self._geometry_

    #def set_geometry(
    #    self,
    #    geometry: Geometry
    #):
    #    self._geometry_ = geometry
    #    return self

    #def set_material(
    #    self,
    #    *,
    #    ambient_strength: float | None = None,
    #    specular_strength: float | None = None,
    #    shininess: float | None = None,
    #    enable_phong_lighting: bool | None = None,
    #    broadcast: bool = True,
    #    type_filter: "type[MeshMobject] | None" = None
    #):
    #    ambient_strength_value = LazyWrapper(ambient_strength) if ambient_strength is not None else None
    #    specular_strength_value = LazyWrapper(specular_strength) if specular_strength is not None else None
    #    shininess_value = LazyWrapper(shininess) if shininess is not None else None
    #    enable_phong_lighting_value = enable_phong_lighting if enable_phong_lighting is not None else \
    #        True if any(param is not None for param in (
    #            ambient_strength,
    #            specular_strength,
    #            shininess
    #        )) else None
    #    if type_filter is None:
    #        type_filter = MeshMobject
    #    for mobject in self.iter_descendants_by_type(mobject_type=type_filter, broadcast=broadcast):
    #        if ambient_strength_value is not None:
    #            mobject._ambient_strength_ = ambient_strength_value
    #        if specular_strength_value is not None:
    #            mobject._specular_strength_ = specular_strength_value
    #        if shininess_value is not None:
    #            mobject._shininess_ = shininess_value
    #        if enable_phong_lighting_value is not None:
    #            mobject._enable_phong_lighting_ = enable_phong_lighting_value
    #    return self

    @property
    def geometry(self) -> Geometry:
        return self._geometry_

    @property
    def color(self) -> Vec3T:
        return self._color_.value

    @property
    def opacity(self) -> float:
        return self._opacity_.value

    @property
    def color_map(self) -> moderngl.Texture | None:
        return self._color_map_.value

    @property
    def enable_phong_lighting(self) -> bool:
        return self._enable_phong_lighting_.value

    @property
    def ambient_strength(self) -> float:
        return self._ambient_strength_.value

    @property
    def specular_strength(self) -> float:
        return self._specular_strength_.value

    @property
    def shininess(self) -> float:
        return self._shininess_.value
