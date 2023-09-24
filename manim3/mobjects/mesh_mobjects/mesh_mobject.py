import moderngl
import numpy as np

from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3f8,
    NP_x3i4
)
from ...lazy.lazy import Lazy
from ...utils.path_utils import PathUtils
from ...rendering.buffers.texture_buffer import TextureBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ..lights.lighting import Lighting
from ..mobject.mobject_attributes.array_attribute import ArrayAttribute
from ..mobject.mobject_attributes.color_attribute import ColorAttribute
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

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _mesh_() -> Mesh:
        return Mesh()

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _color_() -> ColorAttribute:
        return ColorAttribute(np.ones((3,)))

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _opacity_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(1.0)

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _weight_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(1.0)

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _ambient_strength_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(1.0)

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _specular_strength_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(Toplevel.config.mesh_specular_strength)

    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _shininess_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(Toplevel.config.mesh_shininess)

    @Lazy.variable()
    @staticmethod
    def _lighting_() -> Lighting:
        return Toplevel.scene._lighting

    @Lazy.variable_collection()
    @staticmethod
    def _color_maps_() -> tuple[moderngl.Texture, ...]:
        return ()

    @Lazy.property(hasher=Lazy.array_hasher)
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
            fields=[
                "vec3 u_color",
                "float u_opacity",
                "float u_weight",
                "float u_ambient_strength",
                "float u_specular_strength",
                "float u_shininess"
            ],
            data={
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
    def _mesh_vertex_array_(
        color_maps: tuple[moderngl.Texture, ...],
        camera__camera__camera_uniform_block_buffer: UniformBlockBuffer,
        lighting__lighting_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        material_uniform_block_buffer: UniformBlockBuffer,
        mesh__indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_path=PathUtils.shaders_dir.joinpath("mesh.glsl"),
            texture_buffers=[
                TextureBuffer(
                    field="sampler2D t_color_maps[NUM_T_COLOR_MAPS]",
                    array_lens={
                        "NUM_T_COLOR_MAPS": len(color_maps)
                    },
                    texture_array=np.fromiter(color_maps, dtype=moderngl.Texture)
                )
            ],
            uniform_block_buffers=[
                camera__camera__camera_uniform_block_buffer,
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
        self._mesh_vertex_array_.render(target_framebuffer)
