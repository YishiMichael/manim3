__all__ = ["TexturedMeshMobject"]


import moderngl
#import numpy as np
#from trimesh.base import Trimesh

from ..mobjects.mesh_mobject import MeshMobject
from ..utils.context_singleton import ContextSingleton
from ..utils.lazy import lazy_property, lazy_property_initializer
from ..utils.renderable import RenderStep
from ..utils.scene_config import SceneConfig
from ..custom_typing import *


TEXTURED_MESH_VERTEX_SHADER = """
#version 430

in vec3 in_position;
in vec4 in_color;
in vec2 in_uv;

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
    vec2 uv;
} vs_out;

void main() {
    vs_out.color = in_color;
    vs_out.uv = in_uv;
    gl_Position = uniform_projection_matrix * uniform_view_matrix * uniform_model_matrix * uniform_geometry_matrix * vec4(in_position, 1.0);
}
"""

TEXTURED_MESH_FRAGMENT_SHADER = """
#version 430

in VS_FS {
    vec4 color;
    vec2 uv;
} fs_in;

uniform sampler2D uniform_color_map;

out vec4 frag_color;

void main() {
    frag_color = fs_in.color * texture(uniform_color_map, fs_in.uv);
}
"""


class TexturedMeshMobject(MeshMobject):
    @lazy_property
    @classmethod
    def _program_(cls) -> moderngl.Program:
        return ContextSingleton().program(
            vertex_shader=TEXTURED_MESH_VERTEX_SHADER,
            fragment_shader=TEXTURED_MESH_FRAGMENT_SHADER
        )

    #@lazy_property
    #@classmethod
    #def _in_uv_buffer_(cls, geometry: Trimesh) -> moderngl.Buffer:
    #    return cls._make_buffer(geometry.visual.uv.astype(np.float64))

    #@_in_uv_buffer_.releaser
    #@staticmethod
    #def _in_uv_buffer_releaser(in_uv_buffer: moderngl.Buffer) -> None:
    #    in_uv_buffer.release()

    @lazy_property_initializer
    @classmethod
    def _color_map_texture_(cls) -> moderngl.Texture:
        return NotImplemented

    @_color_map_texture_.releaser
    @staticmethod
    def _color_map_texture_releaser(uniform_color_map_texture: moderngl.Texture) -> None:
        uniform_color_map_texture.release()

    def _render(self, scene_config: SceneConfig, target_framebuffer: moderngl.Framebuffer) -> None:
        self._render_by_step(RenderStep(
            program=self._program_,
            index_buffer=self._geometry_._indices_buffer_,
            attributes={
                "in_position": ("3f4 /v", self._geometry_._positions_buffer_),
                "in_uv": ("2f4 /v", self._geometry_._uvs_buffer_),
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
            #        (self._geometry_._uvs_buffer_, "2f8 /v", "in_uv"),
            #        (self._color_buffer_, "4f8 /r", "in_color"),
            #    ],
            #    index_buffer=self._geometry_._indices_buffer_,
            #    mode=moderngl.TRIANGLES
            #),
            textures={
                "uniform_color_map": self._color_map_texture_
            },
            #uniforms={
            #    "uniform_projection_matrix": scene_config._camera_._projection_matrix_buffer_,
            #    "uniform_view_matrix": scene_config._camera_._view_matrix_buffer_,
            #    "uniform_model_matrix": self._model_matrix_buffer_,
            #    "uniform_geometry_matrix": self._geometry_matrix_buffer_,
            #},
            uniform_blocks={
                "uniform_block_camera_matrices": scene_config._camera_._camera_matrices_buffer_,
                "uniform_block_model_matrices": self._model_matrices_buffer_
            },
            subroutines={},
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_,
            mode=moderngl.TRIANGLES
        ))
