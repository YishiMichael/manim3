__all__ = ["MeshMobject"]


import moderngl
import numpy as np

from ..geometries.geometry import Geometry
from ..custom_typing import (
    ColorArrayType,
    Matrix44Type
)
from ..mobjects.mobject import Mobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    RenderStep,
    ShaderStrings,
    TextureStorage,
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig


MESH_VERTEX_SHADER = """
#version 430 core

in vec3 a_position;
in vec2 a_uv;

layout (std140) uniform ub_camera_matrices {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
};
layout (std140) uniform ub_model_matrices {
    mat4 u_model_matrix;
    mat4 u_geometry_matrix;
};

out VS_FS {
    vec2 uv;
} vs_out;

void main() {
    vs_out.uv = a_uv;
    gl_Position = u_projection_matrix * u_view_matrix * u_model_matrix * u_geometry_matrix * vec4(a_position, 1.0);
}
"""

MESH_FRAGMENT_SHADER = """
#version 430
#define NUM_U_COLOR_MAPS _

in VS_FS {
    vec2 uv;
} fs_in;

layout (std140) uniform ub_color {
    vec4 u_color;
};
#if NUM_U_COLOR_MAPS
uniform sampler2D u_color_maps[NUM_U_COLOR_MAPS];
#endif

out vec4 frag_color;

void main() {
    frag_color = u_color;
    #if NUM_U_COLOR_MAPS
    for (int i = 0; i < NUM_U_COLOR_MAPS; ++i) {
        frag_color *= texture(u_color_maps[i], fs_in.uv);
    }
    #endif
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
    @lazy_property
    @staticmethod
    def _geometry_matrix_() -> Matrix44Type:
        return np.identity(4)

    @lazy_property_initializer
    @staticmethod
    def _ub_model_matrices_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer()

    @lazy_property
    @staticmethod
    def _ub_model_matrices_(
        ub_model_matrices_o: UniformBlockBuffer,
        model_matrix: Matrix44Type,
        geometry_matrix: Matrix44Type
    ) -> UniformBlockBuffer:
        ub_model_matrices_o._data_ = [
            (model_matrix, np.float32, None),
            (geometry_matrix, np.float32, None)
        ]
        return ub_model_matrices_o

    @lazy_property_initializer
    @staticmethod
    def _geometry_() -> Geometry:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _color_() -> ColorArrayType:
        return np.ones(4)

    @lazy_property_initializer_writable
    @staticmethod
    def _ub_color_o_() -> UniformBlockBuffer:
        return UniformBlockBuffer()

    @lazy_property
    @staticmethod
    def _ub_color_(
        ub_color_o: UniformBlockBuffer,
        color: ColorArrayType
    ) -> UniformBlockBuffer:
        ub_color_o._data_ = [
            (color, np.float32, None)
        ]
        return ub_color_o

    @lazy_property_initializer
    @staticmethod
    def _color_map_texture_() -> moderngl.Texture | None:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _u_color_maps_o_() -> TextureStorage:
        return TextureStorage()

    @lazy_property
    @staticmethod
    def _u_color_maps_(
        u_color_maps_o: TextureStorage,
        color_map_texture: moderngl.Texture | None
    ) -> TextureStorage:
        textures = [color_map_texture] if color_map_texture is not None else []
        u_color_maps_o._data_ = (textures, "NUM_U_COLOR_MAPS")
        return u_color_maps_o

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.BLEND | moderngl.DEPTH_TEST

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        self._render_by_step(RenderStep(
            shader_strings=ShaderStrings(
                vertex_shader=MESH_VERTEX_SHADER,
                fragment_shader=MESH_FRAGMENT_SHADER
            ),
            texture_storages={
                "u_color_maps": self._u_color_maps_
            },
            uniform_blocks={
                "ub_camera_matrices": scene_config._camera_._ub_camera_matrices_,
                "ub_model_matrices": self._ub_model_matrices_,
                "ub_color": self._ub_color_
            },
            attributes={
                "a_position": self._geometry_._a_position_,
                "a_uv": self._geometry_._a_uv_,
            },
            subroutines={},
            index_buffer=self._geometry_._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            mode=moderngl.TRIANGLES
        ))
