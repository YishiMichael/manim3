__all__ = ["MeshMobject"]


import moderngl
import numpy as np
from trimesh.base import Trimesh
from manim3.utils.context_singleton import ContextSingleton

from manim3.utils.renderable import RenderStep

#from ..cameras.camera import Camera
#from ..geometries.geometry import Geometry
from ..mobjects.mobject import Mobject
from ..mobjects.scene import Scene
from ..utils.lazy import lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..custom_typing import *


MESH_VERTEX_SHADER = """
#version 430

in mat4 in_projection_matrix;
in mat4 in_view_matrix;
in vec3 in_position;
in vec4 in_color;

out VS_FS {
    vec4 color;
} vs_out;

void main() {
    vs_out.color = in_color;
    gl_Position = in_projection_matrix * in_view_matrix * vec4(in_position, 1.0);
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

    @lazy_property_initializer
    @classmethod
    def _geometry_(cls) -> Trimesh:
        return NotImplemented

    @lazy_property_initializer_writable
    @classmethod
    def _color_(cls) -> ColorArrayType:
        return np.ones(4)

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
    def _in_position_buffer_(cls, geometry: Trimesh) -> moderngl.Buffer:
        return cls._make_buffer(geometry.vertices.astype(np.float64))

    @_in_position_buffer_.releaser
    @staticmethod
    def _in_position_buffer_releaser(in_position_buffer: moderngl.Buffer) -> None:
        in_position_buffer.release()

    #@lazy_property
    #@classmethod
    #def _in_uv_buffer_(cls, geometry: Trimesh) -> moderngl.Buffer:
    #    return cls._make_buffer(geometry.visual.uv.astype(np.float64))

    #@_in_uv_buffer_.releaser
    #@staticmethod
    #def _in_uv_buffer_releaser(in_uv_buffer: moderngl.Buffer) -> None:
    #    in_uv_buffer.release()

    @lazy_property
    @classmethod
    def _in_color_buffer_(cls, color: ColorArrayType) -> moderngl.Buffer:
        return cls._make_buffer(color.astype(np.float64))

    @_in_color_buffer_.releaser
    @staticmethod
    def _in_color_buffer_releaser(in_color_buffer: moderngl.Buffer) -> None:
        in_color_buffer.release()

    @lazy_property
    @classmethod
    def _index_buffer_(cls, geometry: Trimesh) -> moderngl.Buffer:
        return cls._make_buffer(geometry.faces.astype(np.int32))

    @_index_buffer_.releaser
    @staticmethod
    def _index_buffer_releaser(index_buffer: moderngl.Buffer) -> None:
        index_buffer.release()

    def _render(self, scene: Scene, target_framebuffer: moderngl.Framebuffer) -> None:
        self._render_by_step(RenderStep(
            vertex_array=ContextSingleton().vertex_array(
                program=self._program_,
                content=[
                    (scene.camera._in_projection_matrix_buffer_, "16f8 /r", "in_projection_matrix"),
                    (scene.camera._in_view_matrix_buffer_, "16f8 /r", "in_view_matrix"),
                    (self._in_position_buffer_, "3f8 /v", "in_position"),
                    #(in_uv_buffer, "2f8 /v", "in_uv"),
                    (self._in_color_buffer_, "4f8 /r", "in_color"),
                ],
                index_buffer=self._index_buffer_,
                mode=moderngl.TRIANGLES
            ),
            textures={},
            uniforms={},
            subroutines={},
            framebuffer=target_framebuffer,
            enable_only=moderngl.BLEND | moderngl.DEPTH_TEST
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
