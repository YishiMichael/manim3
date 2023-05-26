import moderngl
import numpy as np

from ..custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8
)
from ..geometries.geometry import Geometry
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
        interpolate_method=SpaceUtils.lerp_3f8
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register()
    @Lazy.variable_external
    @classmethod
    def _color_map_(cls) -> moderngl.Texture | None:
        return None

    @MobjectStyleMeta.register()
    @Lazy.variable_hashable
    @classmethod
    def _enable_phong_lighting_(cls) -> bool:
        return True

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _ambient_strength_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _specular_strength_(cls) -> NP_f8:
        return 0.5 * np.ones(())  # TODO: config

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp_f8
    )
    @Lazy.variable_array
    @classmethod
    def _shininess_(cls) -> NP_f8:
        return 32.0 * np.ones(())  # TODO: config

    @Lazy.variable
    @classmethod
    def _lighting_uniform_block_buffer_(cls) -> UniformBlockBuffer:
        # Keep updated with `Scene._lighting._lighting_uniform_block_buffer_`.
        return NotImplemented

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(
        cls,
        geometry__position: NP_x3f8
    ) -> NP_x3f8:
        return geometry__position

    @Lazy.property
    @classmethod
    def _material_uniform_block_buffer_(
        cls,
        color: NP_3f8,
        opacity: NP_f8,
        ambient_strength: NP_f8,
        specular_strength: NP_f8,
        shininess: NP_f8
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
                "u_ambient_strength": ambient_strength,
                "u_specular_strength": specular_strength,
                "u_shininess": shininess
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
