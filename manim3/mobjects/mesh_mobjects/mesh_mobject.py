import moderngl
import numpy as np

from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3i4,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.texture_id_buffer import TextureIdBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ..lights.lighting import Lighting
from ..mobject.operation_handlers.lerp_interpolate_handler import LerpInterpolateHandler
from ..mobject.style_meta import StyleMeta
from ..renderable_mobject import RenderableMobject
from .meshes.mesh import Mesh


class MeshMobject(RenderableMobject):
    __slots__ = ()

    def __init__(
        self,
        mesh: Mesh | None = None
    ) -> None:
        super().__init__()
        if mesh is not None:
            self._mesh_ = mesh
        self._lighting_ = Toplevel.scene._lighting

    @StyleMeta.register()
    @Lazy.variable
    @classmethod
    def _mesh_(cls) -> Mesh:
        return Mesh()

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _color_(cls) -> NP_3f8:
        return np.ones((3,))

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _opacity_(cls) -> NP_f8:
        return np.ones(())

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _weight_(cls) -> NP_f8:
        return np.ones(())

    @StyleMeta.register()
    @Lazy.variable
    @classmethod
    def _lighting_(cls) -> Lighting:
        return Lighting()

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _ambient_strength_(cls) -> NP_f8:
        return np.ones(())

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
    )
    @Lazy.variable_array
    @classmethod
    def _specular_strength_(cls) -> NP_f8:
        return Toplevel.config.mesh_specular_strength * np.ones(())

    @StyleMeta.register(
        interpolate_operation=LerpInterpolateHandler
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
    def _local_sample_positions_(
        cls,
        mesh__positions: NP_x3f8,
        mesh__faces: NP_x3i4
    ) -> NP_x3f8:
        return mesh__positions[mesh__faces.flatten()]

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
        mesh__indexed_attributes_buffer: IndexedAttributesBuffer
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
            indexed_attributes_buffer=mesh__indexed_attributes_buffer
        )

    def _render(
        self,
        target_framebuffer: OITFramebuffer
    ) -> None:
        self._mesh_vertex_array_.render(
            framebuffer=target_framebuffer,
            texture_array_dict={
                "t_color_maps": np.fromiter(self._color_maps_, dtype=moderngl.Texture)
            }
        )
