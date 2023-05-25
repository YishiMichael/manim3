import moderngl
import numpy as np

from ..custom_typing import (
    Vec3T,
    Vec3sT
)
from ..geometries.geometry import (
    Geometry,
    GeometryData
)
from ..geometries.plane_geometry import PlaneGeometry
from ..lazy.lazy import Lazy
from ..mobjects.mobject import MobjectStyleMeta
from ..mobjects.renderable_mobject import RenderableMobject
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


class MeshMobject(RenderableMobject):
    __slots__ = ()

    @MobjectStyleMeta.register()
    @Lazy.variable
    @classmethod
    def _geometry_(cls) -> Geometry:
        # Default for `ImageMobject`, `ChildSceneMobject`.
        return PlaneGeometry()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_vec3
    )
    @Lazy.variable_external
    @classmethod
    def _color_(cls) -> Vec3T:
        return np.ones(3)

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _opacity_(cls) -> float:
        return 1.0

    @MobjectStyleMeta.register()
    @Lazy.variable_external
    @classmethod
    def _color_map_(cls) -> moderngl.Texture | None:
        return None

    @MobjectStyleMeta.register()
    @Lazy.variable_shared
    @classmethod
    def _enable_phong_lighting_(cls) -> bool:
        return True

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _ambient_strength_(cls) -> float:
        return 1.0

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _specular_strength_(cls) -> float:
        return 0.5

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_float
    )
    @Lazy.variable_external
    @classmethod
    def _shininess_(cls) -> float:
        return 32.0

    @Lazy.variable
    @classmethod
    def _lighting_uniform_block_buffer_(cls) -> UniformBlockBuffer:
        # Keep updated with `Scene._lighting._lighting_uniform_block_buffer_`.
        return NotImplemented

    @Lazy.property_external
    @classmethod
    def _local_sample_points_(
        cls,
        geometry__geometry_data: GeometryData
    ) -> Vec3sT:
        return geometry__geometry_data.position

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
        camera_uniform_block_buffer: UniformBlockBuffer,
        lighting_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        material_uniform_block_buffer: UniformBlockBuffer,
        geometry__indexed_attributes_buffer: IndexedAttributesBuffer
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
                camera_uniform_block_buffer,
                lighting_uniform_block_buffer,
                model_uniform_block_buffer,
                material_uniform_block_buffer
            ],
            indexed_attributes_buffer=geometry__indexed_attributes_buffer
        )

    def _render(
        self,
        target_framebuffer: OpaqueFramebuffer | TransparentFramebuffer
    ) -> None:
        textures: list[moderngl.Texture] = []
        if (color_map := self._color_map_) is not None:
            textures.append(color_map)
        self._mesh_vertex_array_.render(
            framebuffer=target_framebuffer,
            texture_array_dict={
                "t_color_maps": np.array(textures, dtype=moderngl.Texture)
            }
        )

    #@property
    #def geometry(self) -> Geometry:
    #    return self._geometry_

    #@property
    #def color(self) -> Vec3T:
    #    return self._color_

    #@property
    #def opacity(self) -> float:
    #    return self._opacity_

    #@property
    #def color_map(self) -> moderngl.Texture | None:
    #    return self._color_map_

    #@property
    #def enable_phong_lighting(self) -> bool:
    #    return self._enable_phong_lighting_

    #@property
    #def ambient_strength(self) -> float:
    #    return self._ambient_strength_

    #@property
    #def specular_strength(self) -> float:
    #    return self._specular_strength_

    #@property
    #def shininess(self) -> float:
    #    return self._shininess_
