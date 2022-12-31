__all__ = ["MeshMobject"]


import moderngl
import numpy as np
#from trimesh.base import Trimesh

#from ..cameras.camera import Camera
from ..geometries.geometry import Geometry
from ..mobjects.mobject import Mobject
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..utils.renderable import RenderStep
from ..utils.scene_config import SceneConfig
from ..custom_typing import *


MESH_VERTEX_SHADER = """
#version 430

in vec3 in_position;
in vec4 in_color;

layout (std140) uniform uniform_block_camera_matrices {
    mat4 uniform_projection_matrix;
    mat4 uniform_view_matrix;
};
layout (std140) uniform uniform_block_model_matrices {
    mat4 uniform_model_matrix;
    mat4 uniform_geometry_matrix;
};

out VS_FS {
    vec4 color;
} vs_out;

void main() {
    vs_out.color = in_color;
    gl_Position = uniform_projection_matrix * uniform_view_matrix * uniform_model_matrix * uniform_geometry_matrix * vec4(in_position, 1.0);
}
"""

MESH_FRAGMENT_SHADER = """
#version 430

in VS_FS {
    vec4 color;
} fs_in;

out vec4 frag_color;

void main() {
    frag_color = fs_in.color;
}
"""


#class MeshMaterial(ABC):
#    @abstractmethod
#    def _get_render_step(
#        self,
#        scene: Scene,
#        geometry: Trimesh,
#        target_framebuffer: moderngl.Framebuffer
#    ) -> RenderStep:
#        pass


#class SimpleMeshMaterial(MeshMaterial):
#    def __init__(self, color: ColorArrayType):
#        self.color: ColorArrayType = color


#class TexturedMeshMaterial(MeshMaterial):
#    def __init__(self, color_map: ColorArrayType):
#        self.color: ColorArrayType = color
#    color: ColorArrayType
#    color_map: moderngl.Texture | None


class MeshMobject(Mobject):
    #def __init__(self):
    #    super().__init__()
    #    #self.geometry: Geometry | None = self.init_geometry()  # TODO: consider using trimesh
    #    #self.material: MeshMaterialAttributes = MeshMaterialAttributes(
    #    #    color=np.ones(4),
    #    #    color_map=None
    #    #)

    #@lazy_property_initializer_writable
    #@classmethod
    #def _invisible_(cls) -> bool:
    #    return False

    @lazy_property
    @classmethod
    def _geometry_matrix_(cls) -> Matrix44Type:
        return np.identity(4, dtype=np.float32)

    @lazy_property
    @classmethod
    def _model_matrices_buffer_(cls, model_matrix: Matrix44Type, geometry_matrix: Matrix44Type) -> moderngl.Buffer:
        buffer = ContextSingleton().buffer(reserve=128)
        buffer.write(model_matrix.tobytes(), offset=0)
        buffer.write(geometry_matrix.tobytes(), offset=64)
        return buffer

    @_model_matrices_buffer_.releaser
    @staticmethod
    def _model_matrices_buffer_releaser(model_matrices_buffer: moderngl.Buffer) -> None:
        model_matrices_buffer.release()

    @lazy_property_initializer
    @classmethod
    def _geometry_(cls) -> Geometry:
        return NotImplemented

    @lazy_property_initializer_writable
    @classmethod
    def _color_(cls) -> ColorArrayType:
        return np.ones(4, dtype=np.float32)

    @lazy_property_initializer_writable
    @classmethod
    def _enable_only_(cls) -> int:
        return moderngl.BLEND | moderngl.DEPTH_TEST

    #@lazy_property_initializer
    #@classmethod
    #def _material_(cls) -> MeshMaterial:
    #    return MeshMaterial(
    #        color=np.ones(4),
    #        color_map=None
    #    )

    #@lazy_property_initializer
    #@staticmethod
    #def _color_map_() -> moderngl.Texture | None:
    #    return None

    @lazy_property
    @classmethod
    def _program_(cls) -> moderngl.Program:
        return ContextSingleton().program(
            vertex_shader=MESH_VERTEX_SHADER,
            fragment_shader=MESH_FRAGMENT_SHADER
        )

    @_program_.releaser
    @staticmethod
    def _program_releaser(program: moderngl.Program) -> None:
        program.release()

    @lazy_property
    @classmethod
    def _color_buffer_(cls, color: ColorArrayType) -> moderngl.Buffer:
        return cls._make_buffer(color.astype(np.float64))

    @_color_buffer_.releaser
    @staticmethod
    def _color_buffer_releaser(color_buffer: moderngl.Buffer) -> None:
        color_buffer.release()

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        self._render_by_step(RenderStep(
            program=self._program_,
            index_buffer=self._geometry_._indices_buffer_,
            attributes={
                "in_position": ("3f4 /v", self._geometry_._positions_buffer_),
                "in_color": ("4f4 /r", self._color_buffer_),
            },
            #vertex_array=ContextSingleton().vertex_array(
            #    program=self._program_,
            #    content=[
            #        (scene_config._camera_._projection_matrix_buffer_, "16f8 /r", "in_projection_matrix"),
            #        (scene_config._camera_._view_matrix_buffer_, "16f8 /r", "in_view_matrix"),
            #        (self._model_matrix_buffer_, "16f8 /r", "in_model_matrix"),
            #        (self._geometry_matrix_buffer_, "16f8 /r", "in_geometry_matrix"),
            #        (self._geometry_._positions_buffer_, "3f8 /v", "in_position"),
            #        #(in_uv_buffer, "2f8 /v", "in_uv"),
            #        (self._color_buffer_, "4f8 /r", "in_color"),
            #    ],
            #    index_buffer=self._geometry_._indices_buffer_,
            #    mode=moderngl.TRIANGLES
            #),
            textures={},
            uniform_blocks={
                "uniform_block_camera_matrices": scene_config._camera_._camera_matrices_buffer_,
                "uniform_block_model_matrices": self._model_matrices_buffer_
            },
            subroutines={},
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            mode=moderngl.TRIANGLES
        ))

    #@lazy_property
    #@classmethod
    #def _buffers_from_camera_(
    #    cls,
    #    camera: Camera
    #) -> dict[str, tuple[moderngl.Buffer, str]]:
    #    return {
    #        "in_projection_matrix": (cls._make_buffer(camera.get_projection_matrix()), "16f8 /r"),
    #        "in_view_matrix": (cls._make_buffer(camera.get_view_matrix()), "16f8 /r")
    #    }

    #@lazy_property
    #@classmethod
    #def _buffers_from_geometry_(
    #    cls,
    #    geometry: Trimesh,
    #    define_macros: list[str]
    #) -> dict[str, tuple[moderngl.Buffer, str]]:
    #    buffers = {
    #        "in_position": (cls._make_buffer(geometry.positions), "3f8 /v")
    #    }
    #    if "USE_UV" in define_macros:
    #        buffers["in_uv"] = (cls._make_buffer(geometry.uvs), "2f8 /v")
    #    return buffers

    #@lazy_property
    #@classmethod
    #def _buffers_from_matrix_(
    #    cls,
    #    composite_matrix: Matrix44Type
    #) -> dict[str, tuple[moderngl.Buffer, str]]:
    #    return {
    #        "in_model_matrix": (cls._make_buffer(composite_matrix), "16f8 /r")
    #    }

    #@lazy_property
    #@classmethod
    #def _buffers_from_material_(cls) -> dict[str, tuple[moderngl.Buffer, str]]:  # TODO
    #    color = np.ones(4)
    #    return {
    #        "in_color": (cls._make_buffer(color), "4f8 /r")
    #    }

    #@lazy_property
    #@classmethod
    #def _shader_filename_(cls) -> str:
    #    return "mesh"

    #@lazy_property
    #@classmethod
    #def _define_macros_(cls, color_map: moderngl.Texture | None) -> list[str]:
    #    defines = []
    #    if color_map is not None:
    #        defines.append("USE_UV")
    #        defines.append("USE_COLOR_MAP")
    #    return defines

    #@lazy_property
    #@classmethod
    #def _textures_dict_(
    #    cls,
    #    define_macros: list[str],
    #    color_map: moderngl.Texture | None
    #) -> dict[str, tuple[moderngl.Texture, int]]:
    #    textures = {}
    #    if "USE_COLOR_MAP" in define_macros:
    #        textures["uniform_color_map"] = (color_map, 0)
    #    return textures

    #@lazy_property
    #@classmethod
    #def _buffers_dict_(
    #    cls,
    #    buffers_from_camera: dict[str, tuple[moderngl.Buffer, str]],
    #    buffers_from_geometry: dict[str, tuple[moderngl.Buffer, str]],
    #    #buffers_from_matrix: dict[str, tuple[moderngl.Buffer, str]],
    #    buffers_from_material: dict[str, tuple[moderngl.Buffer, str]],
    #) -> dict[str, tuple[moderngl.Buffer, str]]:
    #    # Update distributively, as making buffers is expensive
    #    return {
    #        **buffers_from_camera,
    #        **buffers_from_geometry,
    #        #**buffers_from_matrix,
    #        **buffers_from_material,
    #    }

    #@lazy_property
    #@classmethod
    #def _vertex_indices_(cls, geometry: Trimesh) -> VertexIndicesType:
    #    return geometry.faces.flatten()

    #@lazy_property
    #@classmethod
    #def _render_primitive_(cls) -> int:
    #    return moderngl.TRIANGLES
