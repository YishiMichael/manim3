import moderngl
import numpy as np

from manim3.mobjects.mobject.mobject_attributes.array_attribute import ArrayAttribute
from manim3.mobjects.mobject.mobject_attributes.color_attribute import ColorAttribute

from ...constants.custom_typing import (
    NP_3f8,
    NP_f8,
    NP_x3i4,
    NP_x3f8
)
from ...lazy.lazy import Lazy
from ...rendering.buffers.texture_buffer import TextureBuffer
from ...rendering.buffers.uniform_block_buffer import UniformBlockBuffer
from ...rendering.framebuffers.oit_framebuffer import OITFramebuffer
from ...rendering.indexed_attributes_buffer import IndexedAttributesBuffer
from ...rendering.vertex_array import VertexArray
from ...toplevel.toplevel import Toplevel
from ..lights.lighting import Lighting
#from ..mobject.operation_handlers.lerp_interpolate_handler import LerpInterpolateHandler
#from ..mobject.style_meta import StyleMeta
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

    #@StyleMeta.register()
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _mesh_() -> Mesh:
        return Mesh()

    #@StyleMeta.register(
    #    interpolate_operation=LerpInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _color_() -> ColorAttribute:
        return ColorAttribute(np.ones((3,)))

    #@StyleMeta.register(
    #    interpolate_operation=LerpInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _opacity_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(1.0)

    #@StyleMeta.register(
    #    interpolate_operation=LerpInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _weight_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(1.0)

    #@StyleMeta.register(
    #    interpolate_operation=LerpInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _ambient_strength_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(1.0)

    #@StyleMeta.register(
    #    interpolate_operation=LerpInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _specular_strength_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(Toplevel.config.mesh_specular_strength)

    #@StyleMeta.register(
    #    interpolate_operation=LerpInterpolateHandler
    #)
    @Lazy.variable(hasher=Lazy.branch_hasher)
    @staticmethod
    def _shininess_() -> ArrayAttribute[NP_f8]:
        return ArrayAttribute(Toplevel.config.mesh_shininess)

    #@StyleMeta.register()
    @Lazy.variable()
    @staticmethod
    def _lighting_() -> Lighting:
        return Lighting()

    @Lazy.variable()
    @staticmethod
    def _color_maps_() -> list[moderngl.Texture]:
        return []

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

    @Lazy.property()
    @staticmethod
    def _mesh_vertex_array_(
        color_maps: list[moderngl.Texture],
        camera__camera_uniform_block_buffer: UniformBlockBuffer,
        lighting__lighting_uniform_block_buffer: UniformBlockBuffer,
        model_uniform_block_buffer: UniformBlockBuffer,
        material_uniform_block_buffer: UniformBlockBuffer,
        mesh__indexed_attributes_buffer: IndexedAttributesBuffer
    ) -> VertexArray:
        return VertexArray(
            shader_filename="mesh",
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
        self._mesh_vertex_array_.render(target_framebuffer)

    def bind_lighting(
        self,
        lighting: Lighting
    ):
        self._lighting_ = lighting
        return self
