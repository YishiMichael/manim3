__all__ = ["Renderable"]


from functools import lru_cache
import os
import re

import moderngl
import skia

from ..utils.context import ContextSingleton
from ..utils.lazy import LazyBase, lazy_property, lazy_property_initializer, lazy_property_initializer_writable
from ..constants import SHADERS_PATH
from ..custom_typing import *


class Renderable(LazyBase):
    @lazy_property_initializer
    @staticmethod
    def _invisible_() -> bool:
        return True

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_depth_test_() -> bool:
        return True

    @lazy_property_initializer_writable
    @staticmethod
    def _enable_blend_() -> bool:
        return True

    @lazy_property_initializer_writable
    @staticmethod
    def _cull_face_() -> str:
        return "back"

    @lazy_property_initializer_writable
    @staticmethod
    def _wireframe_() -> bool:
        return False

    @lazy_property_initializer
    @staticmethod
    def _shader_filename_() -> str:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _define_macros_() -> list[str]:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _textures_dict_() -> dict[str, tuple[moderngl.Texture, int]]:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _buffers_dict_() -> dict[str, tuple[moderngl.Buffer, str]]:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _vertex_indices_() -> VertexIndicesType:
        return NotImplemented

    @lazy_property_initializer
    @staticmethod
    def _render_primitive_() -> int:
        return NotImplemented

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
        define_macros: tuple[str, ...]  # make hashable
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
    @classmethod
    def _program_(
        cls,
        shader_filename: str,
        define_macros: list[str]
    ) -> moderngl.Program:
        return cls._get_program(
            shader_filename,
            tuple(define_macros)
        )

    @lazy_property
    @classmethod
    def _ibo_(cls, vertex_indices: VertexIndicesType) -> moderngl.Buffer:
        return ContextSingleton().buffer(vertex_indices.tobytes())

    @lazy_property
    @classmethod
    def _vao_(
        cls,
        buffers_dict: dict[str, tuple[moderngl.Buffer, str]],
        program: moderngl.Program,
        ibo: moderngl.Buffer,
        render_primitive: int
    ) -> moderngl.VertexArray:
        return ContextSingleton().vertex_array(
            program=program,
            content=[
                (buffer, buffer_format, name)
                for name, (buffer, buffer_format) in buffers_dict.items()
            ],
            index_buffer=ibo,
            index_element_size=4,
            mode=render_primitive
        )

    def render(self) -> None:
        if self._invisible_:
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
            texture.use(location=location)
        vao = self._vao_
        vao.render()
