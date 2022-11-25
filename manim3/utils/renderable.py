from abc import abstractmethod
from functools import lru_cache
import os
import re

import moderngl
import skia

from ..utils.lazy import LazyMeta, lazy_property, lazy_property_initializer
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
#    buffers_dict: dict[str, tuple[AttributeType, str]]
#    vertex_indices: VertexIndicesType
#    render_primitive: int


class Renderable(metaclass=LazyMeta):
    def __init__(self):
        self.invisible: bool = True
        #self.enable_depth_test: bool = True
        #self.enable_blend: bool = True
        #self.cull_face: str = "back"
        #self.wireframe: bool = False

    @lazy_property_initializer
    def _enable_depth_test_() -> bool:
        return True

    @lazy_property_initializer
    def _enable_blend_() -> bool:
        return True

    @lazy_property_initializer
    def _cull_face_() -> str:
        return "back"

    @lazy_property_initializer
    def _wireframe_() -> bool:
        return False

    @lazy_property_initializer
    @abstractmethod
    def _shader_filename_() -> str:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _define_macros_() -> list[str]:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _textures_dict_() -> dict[str, tuple[moderngl.Texture, int]]:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _buffers_dict_() -> dict[str, tuple[moderngl.Buffer, str]]:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _vertex_indices_() -> VertexIndicesType:
        raise NotImplementedError

    @lazy_property_initializer
    @abstractmethod
    def _render_primitive_() -> int:
        raise NotImplementedError

    @classmethod
    def _make_texture(cls, image: skia.Image) -> moderngl.Texture:
        return ContextSingleton().texture(
            size=(image.width(), image.height()),
            components=image.imageInfo().bytesPerPixel(),
            data=image.tobytes(),
        )

    @classmethod
    def _make_buffer(cls, array: AttributeType) -> moderngl.Buffer:
        return ContextSingleton().buffer(array.tobytes())

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
    def _program_(
        cls,
        shader_filename: str,
        define_macros: list[str]
    ) -> moderngl.Program:
        return cls._get_program(
            shader_filename,
            tuple(define_macros)
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
    #    buffers_dict: dict[str, tuple[AttributeType, str]],
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
        #    for name, (array, buffer_format) in self._buffers_dict.items()
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

    #@lazy_property
    #def _raw_textures_dict_(
    #    textures_dict: dict[str, tuple[skia.Image, int]],
    #    program: moderngl.Context
    #) -> dict[int, moderngl.Texture]:
    #    raw_textures = {}
    #    for name, (image, location) in textures_dict.items():
    #        uniform = program.__getitem__(name)
    #        if not isinstance(uniform, moderngl.Uniform):
    #            continue
    #        uniform.__setattr__("value", location)
    #        texture = ContextSingleton().texture(
    #            size=(image.width(), image.height()),
    #            components=image.imageInfo().bytesPerPixel(),
    #            data=image.tobytes(),
    #        )
    #        raw_textures[location] = texture
    #        #texture.use(location=location)
    #        #textures.append(texture)
    #    return raw_textures

    @lazy_property
    def _ibo_(cls, vertex_indices: VertexIndicesType) -> moderngl.Buffer:
        return ContextSingleton().buffer(vertex_indices.tobytes())

    @lazy_property
    def _vao_(
        cls,
        buffers_dict: dict[str, tuple[moderngl.Buffer, str]],
        program: moderngl.Program,
        ibo: moderngl.Buffer,
        render_primitive: int
    ) -> moderngl.VertexArray:
        #program = self._get_program(
        #    ContextSingleton(),
        #    self._shader_filename,
        #    tuple(self._define_macros)  # to hashable
        #)
        #ibo = ctx.buffer(vertex_indices.tobytes())
        return ContextSingleton().vertex_array(
            program=program,
            content=[
                (buffer, buffer_format, name)
                for name, (buffer, buffer_format) in buffers_dict.items()  # TODO: lazify each buffer
            ],
            index_buffer=ibo,
            index_element_size=4,
            mode=render_primitive
        )
        #return self._get_vao(
        #    ContextSingleton(),
        #    program,
        #    self._textures_dict,
        #    #shader_data.uniforms_dict,
        #    self._buffers_dict,
        #    self._vertex_indices,
        #    self._render_primitive
        #)

    def render(self) -> None:
        if self.invisible:
            return

        ctx = ContextSingleton()
        if self._enable_depth_test_:
            ctx.enable(moderngl.DEPTH_TEST)
        else:
            ctx.disable(moderngl.DEPTH_TEST)
        if self._enable_blend_:
            ctx.enable(moderngl.BLEND)
        else:
            ctx.disable(moderngl.BLEND)
        ctx.cull_face = self._cull_face_
        ctx.wireframe = self._wireframe_

        for name, (texture, location) in self._textures_dict_.items():
            uniform = self._program_.__getitem__(name)
            if not isinstance(uniform, moderngl.Uniform):
                continue
            uniform.__setattr__("value", location)
            #texture = ContextSingleton().texture(
            #    size=(image.width(), image.height()),
            #    components=image.imageInfo().bytesPerPixel(),
            #    data=image.tobytes(),
            #)
            texture.use(location=location)
            #raw_textures[location] = texture
            #texture.use(location=location)
            #textures.append(texture)
        vao = self._vao_
        vao.render()
