from dataclasses import dataclass
import re

import moderngl

from utils.typing import *


@dataclass
class ShaderData:
    shader_filename: str
    define_macros: list[str]
    uniforms: dict[str, UniformType]
    texture_arrays: list[TextureArrayType]
    vertex_attributes: AttributesType
    vertex_indices: VertexIndicesType
    render_primitive: int
    enable_depth_test: bool
    enable_blend: bool


class ContextWrapper:
    _GLSL_FILE_CACHE: dict[str, str] = {}

    def __init__(self: Self, ctx: moderngl.Context):
        self._ctx = ctx

    @classmethod
    def _read_glsl_file(cls, filename: str) -> str:
        if filename in cls._GLSL_FILE_CACHE:
            return cls._GLSL_FILE_CACHE[filename]
        with open(f"E:\\ManimKindergarten\\manim3\\src\\shaders\\{filename}.glsl") as f:  # TODO
            result = f.read()
        cls._GLSL_FILE_CACHE[filename] = result
        return result

    @classmethod
    def _insert_defines(cls, content: str, define_macros: list[str]):
        version_str, rest = content.split("\n", 1)
        return "\n".join([
            version_str,
            *[
                f"#define {define_macro}"
                for define_macro in define_macros
            ],
            rest
        ])

    @classmethod
    def _get_program(
        cls,
        ctx: moderngl.Context,
        shader_filename: str,
        define_macros: list[str]
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
        uniforms: dict[str, UniformType],
        texture_arrays: list[TextureArrayType],
        vertex_attributes: AttributesType,
        vertex_indices: VertexIndicesType
    ) -> moderngl.VertexArray:
        for uniform_name, uniform_val in uniforms.items():
            uniform = program[uniform_name]
            if not isinstance(uniform, moderngl.Uniform):
                continue
            if not isinstance(uniform_val, (int, float)):
                uniform_val = tuple(uniform_val.flatten())
            uniform.__setattr__("value", uniform_val)

        for i, texture_array in enumerate(texture_arrays):
            shape = texture_array.shape
            texture = ctx.texture(
                size=shape[:2],
                components=shape[2],
                data=texture_array.tobytes(),
            )
            texture.use(location=i)

        vertex_attribute_names = vertex_attributes.dtype.names
        assert vertex_attribute_names is not None
        #content = [
        #    (
        #        ctx.buffer(vertex_attributes[attribute_name].tobytes()),
        #        moderngl.detect_format(program, [attribute_name]),
        #        attribute_name
        #    )
        #    for attribute_name in vertex_attribute_names
        #]

        vao = ctx.vertex_array(
            program,
            ctx.buffer(vertex_attributes.tobytes()),
            *vertex_attribute_names,
            index_buffer=ctx.buffer(vertex_indices.tobytes()),
            #index_element_size=4,
        )
        return vao

    @classmethod
    def _render(
        cls,
        ctx: moderngl.Context,
        shader_data: ShaderData
    ) -> None:
        program = cls._get_program(
            ctx,
            shader_data.shader_filename,
            shader_data.define_macros
        )
        vao = cls._get_vao(
            ctx,
            program,
            shader_data.uniforms,
            shader_data.texture_arrays,
            shader_data.vertex_attributes,
            shader_data.vertex_indices
        )
        if shader_data.enable_depth_test:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        if shader_data.enable_blend:
            ctx.enable(moderngl.BLEND)
        else:
            ctx.disable(moderngl.BLEND)
        ctx.cull_face = "front_and_back"  # TODO
        vao.render(shader_data.render_primitive)

    def render(self: Self, shader_data: ShaderData) -> None:
        self._render(self._ctx, shader_data)
