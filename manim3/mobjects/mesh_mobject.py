__all__ = ["MeshMobject"]


import moderngl
import numpy as np

from ..geometries.geometry import Geometry
from ..custom_typing import Matrix44Type
from ..mobjects.mobject import Mobject
from ..utils.lazy import (
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)
from ..utils.renderable import (
    Framebuffer,
    RenderStep,
    ShaderStrings,
    TextureStorage,
    UniformBlockBuffer
)
from ..utils.scene_config import SceneConfig


MESH_VERTEX_SHADER = """
#version 430 core

layout (std140) uniform ub_camera_matrices {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
};
layout (std140) uniform ub_model_matrices {
    mat4 u_model_matrix;
    mat4 u_geometry_matrix;
};

in vec3 a_position;
in vec3 a_normal;
in vec2 a_uv;
in vec4 a_color;

out VS_FS {
    vec3 world_position;
    vec3 world_normal;
    vec2 uv;
    vec4 color;
} vs_out;

void main() {
    vs_out.uv = a_uv;
    vs_out.color = a_color;
    vs_out.world_position = vec3(u_model_matrix * u_geometry_matrix * vec4(a_position, 1.0));
    vs_out.world_normal = mat3(transpose(inverse(u_model_matrix * u_geometry_matrix))) * a_normal;
    gl_Position = u_projection_matrix * u_view_matrix * vec4(vs_out.world_position, 1.0);
}
"""

MESH_FRAGMENT_SHADER = """
#version 430

#if NUM_U_COLOR_MAPS
uniform sampler2D u_color_maps[NUM_U_COLOR_MAPS];
#endif

layout (std140) uniform ub_camera_matrices {
    mat4 u_projection_matrix;
    mat4 u_view_matrix;
    vec3 u_view_position;
};
layout (std140) uniform ub_lights {
    vec4 u_ambient_light_color;
    #if NUM_U_POINT_LIGHT_POSITIONS
    vec3 u_point_light_positions[NUM_U_POINT_LIGHT_POSITIONS];
    #endif
    #if NUM_U_POINT_LIGHT_COLORS
    vec4 u_point_light_colors[NUM_U_POINT_LIGHT_COLORS];
    #endif
};

in VS_FS {
    vec3 world_position;
    vec3 world_normal;
    vec2 uv;
    vec4 color;
} fs_in;

out vec4 frag_color;

void main() {
    frag_color = vec4(0.0);

    frag_color += u_ambient_light_color;

    vec3 normal = normalize(fs_in.world_normal);
    #if NUM_U_POINT_LIGHT_COLORS
    for (int i = 0; i < NUM_U_POINT_LIGHT_COLORS; ++i) {
        vec3 light_direction = normalize(u_point_light_positions[i] - fs_in.world_position);
        vec4 light_color = u_point_light_colors[i];

        vec4 diffuse = max(dot(normal, light_direction), 0.0) * light_color;
        frag_color += diffuse;

        vec3 view_direction = normalize(u_view_position - fs_in.world_position);
        vec3 reflect_direction = reflect(-light_direction, normal);
        vec4 specular = 0.5 * pow(max(dot(view_direction, reflect_direction), 0.0), 32) * light_color;
        frag_color += specular;
    }
    #endif

    frag_color *= fs_in.color;
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
        return UniformBlockBuffer({
            "u_model_matrix": "mat4",
            "u_geometry_matrix": "mat4"
        })

    @lazy_property
    @staticmethod
    def _ub_model_matrices_(
        ub_model_matrices_o: UniformBlockBuffer,
        model_matrix: Matrix44Type,
        geometry_matrix: Matrix44Type
    ) -> UniformBlockBuffer:
        ub_model_matrices_o.write({
            "u_model_matrix": model_matrix,
            "u_geometry_matrix": geometry_matrix
        })
        return ub_model_matrices_o

    @lazy_property_initializer_writable
    @staticmethod
    def _geometry_() -> Geometry:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _color_map_texture_() -> moderngl.Texture | None:
        return None

    @lazy_property_initializer
    @staticmethod
    def _u_color_maps_o_() -> TextureStorage:
        return TextureStorage("sampler2D[NUM_U_COLOR_MAPS]")

    @lazy_property
    @staticmethod
    def _u_color_maps_(
        u_color_maps_o: TextureStorage,
        color_map_texture: moderngl.Texture | None
    ) -> TextureStorage:
        textures = [color_map_texture] if color_map_texture is not None else []
        u_color_maps_o.write(textures)
        return u_color_maps_o

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_only_() -> int:
        return moderngl.BLEND | moderngl.DEPTH_TEST

    def _render(self, scene_config: SceneConfig, target_framebuffer: Framebuffer) -> None:
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
                "ub_lights": scene_config._ub_lights_
                #"ub_color": self._ub_color_
            },
            attributes={
                "a_position": self._geometry_._a_position_,
                "a_normal": self._geometry_._a_normal_,
                "a_uv": self._geometry_._a_uv_,
                "a_color": self._geometry_._a_color_
            },
            subroutines={},
            index_buffer=self._geometry_._index_buffer_,
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            mode=moderngl.TRIANGLES
        ))
