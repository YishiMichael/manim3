import moderngl
import numpy as np

from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8,
    NP_xi4
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.texture_id_buffer import TextureIdBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ...utils.space import SpaceUtils
from ..lights.lighting import Lighting
from ..mobject.mobject_style_meta import MobjectStyleMeta
from ..renderable_mobject import RenderableMobject
from .meshes.mesh import Mesh
from .meshes.plane_mesh import PlaneMesh


class MeshMobject(RenderableMobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()
        self._lighting_ = Toplevel.scene._lighting

    @MobjectStyleMeta.register()
    @Lazy.variable
    @classmethod
    def _mesh_(cls) -> Mesh:
        # Default for `ImageMobject`.
        return PlaneMesh()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return (1.0 - 2 ** (-32)) * np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _weight_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register()
    @Lazy.variable
    @classmethod
    def _lighting_(cls) -> Lighting:
        return Lighting()

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _ambient_strength_(cls) -> NP_f8:
        return np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _specular_strength_(cls) -> NP_f8:
        return Toplevel.config.mesh_specular_strength * np.ones(())

    @MobjectStyleMeta.register(
        interpolate_method=SpaceUtils.lerp
    )
    @Lazy.variable_array
    @classmethod
    def _shininess_(cls) -> NP_f8:
        return Toplevel.config.mesh_shininess * np.ones(())

    @Lazy.variable_external
    @classmethod
    def _color_maps_(cls) -> list[moderngl.Texture]:
        return []

    @Lazy.property_array
    @classmethod
    def _local_sample_points_(
        cls,
        geometry__position: NP_x3f8,
        geometry__index: NP_xi4
    ) -> NP_x3f8:
        return geometry__position[geometry__index]

    @Lazy.property
    @classmethod
    def _material_uniform_block_buffer_(
        cls,
        color: NP_3f8,
        opacity: NP_f8,
        weight: NP_f8,
        ambient_strength: NP_f8,
        specular_strength: NP_f8,
        shininess: NP_f8
    ) -> UniformBlockBuffer:
        return UniformBlockBuffer(
            name="ub_material",
            fields=[
                "vec3 u_color",
                "float u_opacity",
                "float u_weight",
                "float u_ambient_strength",
                "float u_specular_strength",
                "float u_shininess"
            ],
            data={
                "u_color": color,
                "u_opacity": opacity,
                "u_weight": weight,
                "u_ambient_strength": ambient_strength,
                "u_specular_strength": specular_strength,
                "u_shininess": shininess
            }
        )

    @Lazy.property
    @classmethod
    def _mesh_vertex_array_(
        cls,
        color_maps: list[moderngl.Texture],
        camera__camera_uniform_block_buffer: UniformBlockBuffer,
        lighting__lighting_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        material_uniform_block_buffer: UniformBlockBuffer,
        geometry__indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="mesh",
            texture_id_buffers=[
                TextureIdBuffer(
                    field="sampler2D t_color_maps[NUM_T_COLOR_MAPS]",
                    array_lens={
                        "NUM_T_COLOR_MAPS": len(color_maps)
                    }
                )
            ],
            uniform_block_buffers=[
                camera__camera_uniform_block_buffer,
                lighting__lighting_uniform_block_buffer,
                model_uniform_block_buffer,
                material_uniform_block_buffer
            ],
            indexed_attributes_buffer=geometry__indexed_attributes_buffer
        )

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._mesh_vertex_array_.render(
            framebuffer=target_framebuffer,
            texture_array_dict={
                "t_color_maps": np.array(self._color_maps_, dtype=moderngl.Texture)
            }
        )
