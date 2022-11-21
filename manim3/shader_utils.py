from dataclasses import dataclass
from functools import lru_cache
import os
import re
import skia

import moderngl

from .constants import SHADERS_PATH
from .custom_typing import *


@dataclass
class ShaderData:
    enable_depth_test: bool
    enable_blend: bool
    cull_face: str
    wireframe: bool
    shader_filename: str
    define_macros: list[str]
    textures_dict: dict[str, tuple[skia.Pixmap, int]]
    #uniforms_dict: dict[str, UniformType]
    attributes_dict: dict[str, tuple[AttributeType, str]]
    vertex_indices: VertexIndicesType
    render_primitive: int


class ContextWrapper:
    #_GLSL_FILE_CACHE: dict[str, str] = {}
    #_GLSL_PROGRAM_CACHE

    def __init__(self: Self, ctx: moderngl.Context):
        self.ctx: moderngl.Context = ctx

    @classmethod
    @lru_cache(maxsize=8, typed=True)
    def _read_glsl_file(cls, filename: str) -> str:
        #if filename in cls._GLSL_FILE_CACHE:
        #    return cls._GLSL_FILE_CACHE[filename]
        with open(os.path.join(SHADERS_PATH, f"{filename}.glsl")) as f:
            result = f.read()
        #cls._GLSL_FILE_CACHE[filename] = result
        return result

    @classmethod
    def _insert_defines(cls, content: str, define_macros: tuple[str, ...]):
        version_str, rest = content.split("\n", 1)
        return "\n".join([
            version_str,
            *(
                f"#define {define_macro}"
                for define_macro in define_macros
            ),
            rest
        ])

    @classmethod
    @lru_cache(maxsize=8, typed=True)
    def _get_program(
        cls,
        ctx: moderngl.Context,
        shader_filename: str,
        define_macros: tuple[str, ...]
    ) -> moderngl.Program:
        content = cls._insert_defines(cls._read_glsl_file(shader_filename), define_macros)
        shaders_dict = dict.fromkeys((
            "VERTEX_SHADER",
            "FRAGMENT_SHADER",
            "GEOMETRY_SHADER",
            "TESS_CONTROL_SHADER",
            "TESS_EVALUATION_SHADER"
        ))
        for shader_type in shaders_dict:
            if re.search(f"\\b{shader_type}\\b", content):
                shaders_dict[shader_type] = cls._insert_defines(content, [shader_type])
        if shaders_dict["VERTEX_SHADER"] is None:
            raise
        program = ctx.program(
            vertex_shader=shaders_dict["VERTEX_SHADER"],
            fragment_shader=shaders_dict["FRAGMENT_SHADER"],
            geometry_shader=shaders_dict["GEOMETRY_SHADER"],
            tess_control_shader=shaders_dict["TESS_CONTROL_SHADER"],
            tess_evaluation_shader=shaders_dict["TESS_EVALUATION_SHADER"]
        )
        return program

    @classmethod
    def _get_vao(
        cls,
        ctx: moderngl.Context,
        program: moderngl.Program,
        textures_dict: dict[str, tuple[skia.Pixmap, int]],
        #uniforms_dict: dict[str, UniformType],
        attributes_dict: dict[str, tuple[AttributeType, str]],
        vertex_indices: VertexIndicesType,
        render_primitive: int
    ) -> moderngl.VertexArray:
        #for uniform_name, uniform_val in uniforms_dict.items():
        #    uniform = program[uniform_name]
        #    if not isinstance(uniform, moderngl.Uniform):
        #        continue
        #    if not isinstance(uniform_val, (int, float)):
        #        uniform_val = tuple(uniform_val.flatten())
        #    uniform.__setattr__("value", uniform_val)

        for name, (pixmap, location) in textures_dict.items():
            uniform = program[name]
            if not isinstance(uniform, moderngl.Uniform):
                continue
            uniform.__setattr__("value", location)
            texture = ctx.texture(
                size=(pixmap.width(), pixmap.height()),
                components=pixmap.info().bytesPerPixel(),
                data=pixmap,
            )
            texture.use(location=location)

        content = [
            (ctx.buffer(array.tobytes()), buffer_format, name)
            for name, (array, buffer_format) in attributes_dict.items()
        ]
        ibo = ctx.buffer(vertex_indices.tobytes())
        vao = ctx.vertex_array(
            program=program,
            content=content,
            index_buffer=ibo,
            index_element_size=4,
            mode=render_primitive
        )
        return vao

    @classmethod
    def _configure_context(
        cls,
        ctx: moderngl.Context,
        enable_depth_test: bool,
        enable_blend: bool,
        cull_face: str,
        wireframe: bool
    ) -> None:
        if enable_depth_test:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        if enable_blend:
            ctx.enable(moderngl.BLEND)
        else:
            ctx.disable(moderngl.BLEND)
        ctx.cull_face = cull_face
        ctx.wireframe = wireframe

    def render(self: Self, shader_data: ShaderData) -> None:
        #import time
        #t = time.time()
        #print(time.time()-t)
        ctx = self.ctx
        self._configure_context(
            ctx,
            shader_data.enable_depth_test,
            shader_data.enable_blend,
            shader_data.cull_face,
            shader_data.wireframe
        )
        #print(time.time()-t)
        program = self._get_program(
            ctx,
            shader_data.shader_filename,
            tuple(shader_data.define_macros)  # to hashable
        )
        #print(time.time()-t)
        vao = self._get_vao(
            ctx,
            program,
            shader_data.textures_dict,
            #shader_data.uniforms_dict,
            shader_data.attributes_dict,
            shader_data.vertex_indices,
            shader_data.render_primitive
        )
        #print(time.time()-t)
        vao.render()
        #print(time.time()-t)
