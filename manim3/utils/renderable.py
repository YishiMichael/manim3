from abc import abstractmethod
from functools import lru_cache
import os
import re

import moderngl
import skia

from ..utils.lazy import LazyMeta, lazy_property
from ..constants import SHADERS_PATH
from ..custom_typing import *


class ContextSingleton:
    _instance: moderngl.Context | None = None

    def __new__(cls):
        assert cls._instance is not None
        return cls._instance


#@dataclass
#class ShaderData:
#    enable_depth_test: bool
#    enable_blend: bool
#    cull_face: str
#    wireframe: bool
#    shader_filename: str
#    define_macros: list[str]
#    textures_dict: dict[str, tuple[skia.Image, int]]
#    #uniforms_dict: dict[str, UniformType]
#    attributes_dict: dict[str, tuple[AttributeType, str]]
#    vertex_indices: VertexIndicesType
#    render_primitive: int


class Renderable(metaclass=LazyMeta):
    def __init__(self: Self):
        self.abandon_render: bool = True
        self.enable_depth_test: bool = True
        self.enable_blend: bool = True
        self.cull_face: str = "back"
        self.wireframe: bool = False

    @classmethod
    @lru_cache(maxsize=8, typed=True)
    def _read_glsl_file(cls, filename: str) -> str:
        with open(os.path.join(SHADERS_PATH, f"{filename}.glsl")) as f:
            result = f.read()
        return result

    @classmethod
    def _insert_defines(cls, content: str, define_macros: list[str]):
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
        shader_filename: str,
        define_macros: tuple[str, ...]
    ) -> moderngl.Program:
        content = cls._insert_defines(
            cls._read_glsl_file(shader_filename),
            list(define_macros)
        )
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
        return ContextSingleton().program(
            vertex_shader=shaders_dict["VERTEX_SHADER"],
            fragment_shader=shaders_dict["FRAGMENT_SHADER"],
            geometry_shader=shaders_dict["GEOMETRY_SHADER"],
            tess_control_shader=shaders_dict["TESS_CONTROL_SHADER"],
            tess_evaluation_shader=shaders_dict["TESS_EVALUATION_SHADER"]
        )

    @lazy_property
    def _program(self: Self) -> moderngl.Program:
        return self._get_program(
            self._shader_filename,
            tuple(self._define_macros)
        )
        #content = self._insert_defines(
        #    self._read_glsl_file(self._shader_filename),
        #    self._define_macros
        #)
        #shaders_dict = dict.fromkeys((
        #    "VERTEX_SHADER",
        #    "FRAGMENT_SHADER",
        #    "GEOMETRY_SHADER",
        #    "TESS_CONTROL_SHADER",
        #    "TESS_EVALUATION_SHADER"
        #))
        #for shader_type in shaders_dict:
        #    if re.search(f"\\b{shader_type}\\b", content):
        #        shaders_dict[shader_type] = self._insert_defines(content, [shader_type])
        #if shaders_dict["VERTEX_SHADER"] is None:
        #    raise
        #return ContextSingleton().program(
        #    vertex_shader=shaders_dict["VERTEX_SHADER"],
        #    fragment_shader=shaders_dict["FRAGMENT_SHADER"],
        #    geometry_shader=shaders_dict["GEOMETRY_SHADER"],
        #    tess_control_shader=shaders_dict["TESS_CONTROL_SHADER"],
        #    tess_evaluation_shader=shaders_dict["TESS_EVALUATION_SHADER"]
        #)

    #@classmethod
    #def _get_vao(
    #    cls,
    #    ctx: moderngl.Context,
    #    program: moderngl.Program,
    #    textures_dict: dict[str, tuple[skia.Image, int]],
    #    #uniforms_dict: dict[str, UniformType],
    #    attributes_dict: dict[str, tuple[AttributeType, str]],
    #    vertex_indices: VertexIndicesType,
    #    render_primitive: int
    #) -> moderngl.VertexArray:
        #for uniform_name, uniform_val in uniforms_dict.items():
        #    uniform = program[uniform_name]
        #    if not isinstance(uniform, moderngl.Uniform):
        #        continue
        #    if not isinstance(uniform_val, (int, float)):
        #        uniform_val = tuple(uniform_val.flatten())
        #    uniform.__setattr__("value", uniform_val)

        #for name, (image, location) in textures_dict.items():
        #    uniform = program[name]
        #    if not isinstance(uniform, moderngl.Uniform):
        #        continue
        #    uniform.__setattr__("value", location)
        #    texture = ctx.texture(
        #        size=(image.width(), image.height()),
        #        components=image.imageInfo().bytesPerPixel(),
        #        data=image.tobytes(),
        #    )
        #    texture.use(location=location)

        #content = [
        #    (ctx.buffer(array.tobytes()), buffer_format, name)
        #    for name, (array, buffer_format) in self._attributes_dict.items()
        #]
        ##ibo = ctx.buffer(vertex_indices.tobytes())
        #return ctx.vertex_array(
        #    program=program,
        #    content=content,
        #    index_buffer=self._ibo,
        #    index_element_size=4,
        #    mode=self._render_primitive
        #)
        #return vao

    @lazy_property
    def _textures(self: Self) -> list[moderngl.Texture]:
        textures = []
        for name, (image, location) in self._textures_dict.items():
            uniform = self._program[name]
            if not isinstance(uniform, moderngl.Uniform):
                continue
            uniform.__setattr__("value", location)
            texture = ContextSingleton().texture(
                size=(image.width(), image.height()),
                components=image.imageInfo().bytesPerPixel(),
                data=image.tobytes(),
            )
            texture.use(location=location)
            textures.append(texture)
        return textures

    @lazy_property
    def _ibo(self: Self) -> moderngl.Buffer:
        return ContextSingleton().buffer(self._vertex_indices.tobytes())

    @lazy_property
    def _vao(self: Self) -> moderngl.VertexArray:
        #program = self._get_program(
        #    ContextSingleton(),
        #    self._shader_filename,
        #    tuple(self._define_macros)  # to hashable
        #)
        content = [
            (ContextSingleton().buffer(array.tobytes()), buffer_format, name)
            for name, (array, buffer_format) in self._attributes_dict.items()
        ]
        #ibo = ctx.buffer(vertex_indices.tobytes())
        return ContextSingleton().vertex_array(
            program=self._program,
            content=content,
            index_buffer=self._ibo,
            index_element_size=4,
            mode=self._render_primitive
        )
        #return self._get_vao(
        #    ContextSingleton(),
        #    program,
        #    self._textures_dict,
        #    #shader_data.uniforms_dict,
        #    self._attributes_dict,
        #    self._vertex_indices,
        #    self._render_primitive
        #)

    def render(self: Self) -> None:
        if self.abandon_render:
            return

        ctx = ContextSingleton()
        if self._enable_depth_test:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        if self._enable_blend:
            ctx.enable(moderngl.BLEND)
        else:
            ctx.disable(moderngl.BLEND)
        ctx.cull_face = self._cull_face
        ctx.wireframe = self._wireframe
        vao = self._vao
        vao.render()

    @lazy_property
    def _enable_depth_test(self: Self) -> bool:
        return self.enable_depth_test

    @_enable_depth_test.setter
    def _enable_depth_test(self: Self, arg: bool) -> None:
        pass

    @lazy_property
    def _enable_blend(self: Self) -> bool:
        return self.enable_blend

    @_enable_blend.setter
    def _enable_blend(self: Self, arg: bool) -> None:
        pass

    @lazy_property
    def _cull_face(self: Self) -> str:
        return self.cull_face

    @_cull_face.setter
    def _cull_face(self: Self, arg: str) -> None:
        pass

    @lazy_property
    def _wireframe(self: Self) -> bool:
        return self.wireframe

    @_wireframe.setter
    def _wireframe(self: Self, arg: bool) -> None:
        pass

    @lazy_property
    @abstractmethod
    def _shader_filename(self: Self) -> str:
        raise NotImplementedError

    @lazy_property
    @abstractmethod
    def _define_macros(self: Self) -> list[str]:
        raise NotImplementedError

    @lazy_property
    @abstractmethod
    def _textures_dict(self: Self) -> dict[str, tuple[skia.Image, int]]:
        raise NotImplementedError

    @lazy_property
    @abstractmethod
    def _attributes_dict(self: Self) -> dict[str, tuple[AttributeType, str]]:
        raise NotImplementedError

    @lazy_property
    @abstractmethod
    def _vertex_indices(self: Self) -> VertexIndicesType:
        raise NotImplementedError

    @lazy_property
    @abstractmethod
    def _render_primitive(self: Self) -> int:
        raise NotImplementedError
