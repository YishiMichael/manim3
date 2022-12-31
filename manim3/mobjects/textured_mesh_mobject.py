__all__ = ["TexturedMeshMobject"]


import moderngl
import numpy as np
from trimesh.base import Trimesh
from manim3.utils.context_singleton import ContextSingleton

from manim3.utils.renderable import RenderStep

from ..mobjects.mesh_mobject import MeshMobject
from ..mobjects.scene import Scene
from ..utils.lazy import lazy_property, lazy_property_initializer
from ..custom_typing import *


TEXTURED_MESH_VERTEX_SHADER = """
#version 430

in mat4 in_projection_matrix;
in mat4 in_view_matrix;
//in mat4 in_model_matrix;
in vec3 in_position;
in vec4 in_color;
in vec2 in_uv;

out VS_FS {
    vec4 color;
    vec2 uv;
} vs_out;

void main() {
    vs_out.color = in_color;
    vs_out.uv = in_uv;
    gl_Position = in_projection_matrix * in_view_matrix * vec4(in_position, 1.0);
}
"""

TEXTURED_MESH_FRAGMENT_SHADER = """
#version 430

in VS_FS {
    vec4 color;
    in vec2 uv;
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

    @lazy_property
    @classmethod
    def _in_uv_buffer_(cls, geometry: Trimesh) -> moderngl.Buffer:
        return cls._make_buffer(geometry.visual.uv.astype(np.float64))

    @_in_uv_buffer_.releaser
    @staticmethod
    def _in_uv_buffer_releaser(in_uv_buffer: moderngl.Buffer) -> None:
        in_uv_buffer.release()

    @lazy_property_initializer
    @classmethod
    def _uniform_color_map_texture_(cls) -> moderngl.Texture:
        return NotImplemented

    @_uniform_color_map_texture_.releaser
    @staticmethod
    def _uniform_color_map_texture_releaser(uniform_color_map_texture: moderngl.Texture) -> None:
        uniform_color_map_texture.release()

    def _render(self, scene: Scene, target_framebuffer: moderngl.Framebuffer) -> None:
        self._render_by_step(RenderStep(
            vertex_array=ContextSingleton().vertex_array(
                program=self._program_,
                content=[
                    (scene.camera._in_projection_matrix_buffer_, "16f8 /r", "in_projection_matrix"),
                    (scene.camera._in_view_matrix_buffer_, "16f8 /r", "in_view_matrix"),
                    (self._in_position_buffer_, "3f8 /v", "in_position"),
                    (self._in_uv_buffer_, "2f8 /v", "in_uv"),
                    (self._in_color_buffer_, "4f8 /r", "in_color"),
                ],
                index_buffer=self._index_buffer_,
                mode=moderngl.TRIANGLES
            ),
            textures={
                "uniform_color_map": self._uniform_color_map_texture_
            },
            uniforms={},
            subroutines={},
            framebuffer=target_framebuffer,
            enable_only=self._enable_only_
        ))
