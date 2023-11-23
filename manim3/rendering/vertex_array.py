from __future__ import annotations


import itertools
import re
from typing import Self

import attrs
import moderngl

from ..lazy.lazy import Lazy
from ..lazy.lazy_object import LazyObject
from ..toplevel.toplevel import Toplevel
from .buffers.attributes_buffer import AttributesBuffer
from .buffers.texture_buffer import TextureBuffer
from .buffers.uniform_block_buffer import UniformBlockBuffer
from .framebuffers.framebuffer import Framebuffer
from .field import (
    AtomicField,
    StructuredField
)


@attrs.frozen(kw_only=True)
class ProgramUniformInfo:
    array_length: int
    binding: int

    def verify_texture_buffer(
        self: Self,
        texture_buffer: TextureBuffer
    ) -> None:
        assert len(texture_buffer._textures_) == self.array_length


@attrs.frozen(kw_only=True)
class ProgramUniformBlockInfo:
    size: int
    binding: int

    def verify_structured_field(
        self: Self,
        field: StructuredField
    ) -> None:
        assert field._itemsize_ == self.size


@attrs.frozen(kw_only=True)
class ProgramAttributeInfo:
    array_length: int
    dimension: int
    shape: str

    def verify_atomic_field(
        self: Self,
        field: AtomicField
    ) -> None:
        assert field._size_ == self.array_length
        assert field._col_len_ * field._row_len_ == self.dimension
        assert field._base_char_.replace("u", "I") == self.shape


@attrs.frozen(kw_only=True)
class VertexArrayInfo:
    vertex_array: moderngl.VertexArray
    texture_bindings: tuple[tuple[moderngl.Texture, int], ...]
    uniform_block_bindings: tuple[tuple[str, int], ...]


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        shader_filename: str,
        custom_macros: tuple[str, ...] = (),
        texture_buffers: tuple[TextureBuffer, ...] = (),
        uniform_block_buffers: tuple[UniformBlockBuffer, ...] = (),
        attributes_buffer: AttributesBuffer
    ) -> None:
        super().__init__()
        self._shader_filename_ = shader_filename
        self._custom_macros_ = custom_macros
        self._texture_buffers_ = texture_buffers
        self._uniform_block_buffers_ = uniform_block_buffers
        self._attributes_buffer_ = attributes_buffer

    @Lazy.variable()
    @staticmethod
    def _shader_filename_() -> str:
        return ""

    @Lazy.variable(plural=True)
    @staticmethod
    def _custom_macros_() -> tuple[str, ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _texture_buffers_() -> tuple[TextureBuffer, ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _uniform_block_buffers_() -> tuple[UniformBlockBuffer, ...]:
        return ()

    @Lazy.variable()
    @staticmethod
    def _attributes_buffer_() -> AttributesBuffer:
        return NotImplemented

    @Lazy.property(plural=True)
    @staticmethod
    def _macros_(
        custom_macros: tuple[str, ...],
        texture_buffers__macros: tuple[tuple[str, ...], ...],
        uniform_block_buffers__macros: tuple[tuple[str, ...], ...],
        attributes_buffer__macros: tuple[str, ...]
    ) -> tuple[str, ...]:
        return tuple(itertools.chain(
            custom_macros,
            itertools.chain.from_iterable(texture_buffers__macros),
            itertools.chain.from_iterable(uniform_block_buffers__macros),
            attributes_buffer__macros
        ))

    @Lazy.property()
    @staticmethod
    def _program_(
        shader_filename: str,
        macros: tuple[str, ...],
    ) -> moderngl.Program:

        def read_shader_with_includes_replaced(
            shader_filename: str
        ) -> str:
            for shader_dir in Toplevel._get_config().shader_search_dirs:
                if (shader_path := shader_dir.joinpath(shader_filename)).exists():
                    break
            else:
                raise FileNotFoundError(shader_filename)

            shader_text = shader_path.read_text(encoding="utf-8")
            return re.sub(
                r"#include \"(.+?)\"",
                lambda match: read_shader_with_includes_replaced(match.group(1)),
                shader_text
            )

        shader_text = read_shader_with_includes_replaced(shader_filename)
        shaders = {
            shader_type: "\n".join((
                f"#version {Toplevel._get_context().version_code} core",
                "\n",
                f"#define {shader_type}",
                *macros,
                "\n",
                shader_text
            ))
            for shader_type in (
                "VERTEX_SHADER",
                "FRAGMENT_SHADER",
                "GEOMETRY_SHADER",
                "TESS_CONTROL_SHADER",
                "TESS_EVALUATION_SHADER"
            )
            if re.search(rf"\b{shader_type}\b", shader_text, flags=re.MULTILINE) is not None
        }
        return Toplevel._get_context().program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER")
        )

    @Lazy.property()
    @staticmethod
    def _vertex_array_info_(
        program: moderngl.Program,
        texture_buffers: tuple[TextureBuffer, ...],
        uniform_block_buffers__field: tuple[StructuredField, ...],
        attributes_buffer: AttributesBuffer
    ) -> VertexArrayInfo | None:
        uniform_info_dict: dict[str, ProgramUniformInfo] = {}
        uniform_block_info_dict: dict[str, ProgramUniformBlockInfo] = {}
        attribute_info_dict: dict[str, ProgramAttributeInfo] = {}
        texture_binding = 1
        uniform_block_binding = 0
        for name in program:
            assert re.fullmatch(r"\w+", name)

            match program[name]:
                case moderngl.Uniform() as uniform:
                    # Used as a `sampler2D`.
                    assert uniform.dimension == 1
                    array_length = uniform.array_length
                    uniform_info_dict[name] = ProgramUniformInfo(
                        array_length=array_length,
                        binding=texture_binding
                    )
                    uniform.value = (
                        texture_binding
                        if array_length == 1
                        else range(texture_binding, texture_binding + array_length)
                    )
                    texture_binding += array_length

                case moderngl.UniformBlock() as uniform_block:
                    uniform_block_info_dict[name] = ProgramUniformBlockInfo(
                        size=uniform_block.size,
                        binding=uniform_block_binding
                    )
                    uniform_block.binding = uniform_block_binding
                    uniform_block_binding += 1

                case moderngl.Attribute() as attribute:
                    attribute_info_dict[name] = ProgramAttributeInfo(
                        array_length=attribute.array_length,
                        dimension=attribute.dimension,
                        shape=attribute.shape
                    )

        texture_bindings: list[tuple[moderngl.Texture, int]] = []
        for texture_buffer in texture_buffers:
            if (uniform_info := uniform_info_dict.pop(texture_buffer._name_, None)) is None:
                continue
            uniform_info.verify_texture_buffer(texture_buffer)
            texture_bindings.extend(
                (texture, binding)
                for binding, texture in enumerate(texture_buffer._textures_, start=uniform_info.binding)
            )

        uniform_block_bindings: list[tuple[str, int]] = []
        for field in uniform_block_buffers__field:
            if (uniform_block_info := uniform_block_info_dict.pop(field._name_, None)) is None:
                continue
            uniform_block_info.verify_structured_field(field)
            uniform_block_bindings.append((field._name_, uniform_block_info.binding))

        attribute_names: list[str] = []
        format_components: list[str] = []
        for field, padding in zip(attributes_buffer._fields_, attributes_buffer._merged_field_._paddings_, strict=True):
            if (attribute_info := attribute_info_dict.pop(field._name_, None)) is None:
                if (total_padding := field._itemsize_ * field._size_ + padding):
                    format_components.append(f"{total_padding}x")
                continue
            attribute_info.verify_atomic_field(field)
            attribute_names.append(field._name_)
            for _ in range(field._size_):
                for _ in range(field._row_len_):
                    format_components.append(f"{field._col_len_}{field._base_char_}{field._base_itemsize_}")
                    if (col_padding := field._col_padding_):
                        format_components.append(f"{col_padding}x{field._base_itemsize_}")
            if padding:
                format_components.append(f"{padding}x")
        format_components.append("/v")

        assert not uniform_info_dict
        assert not uniform_block_info_dict
        assert not attribute_info_dict

        if (
            not attributes_buffer._vertices_count_
            or not attributes_buffer._merged_field_._itemsize_
            or attributes_buffer._use_index_buffer_ and not attributes_buffer._index_bytes_
        ):
            return None
        return VertexArrayInfo(
            vertex_array=Toplevel._get_context().vertex_array(
                program=program,
                attributes_buffer=attributes_buffer._buffer_,
                attributes_buffer_format_str=" ".join(format_components),
                attribute_names=tuple(attribute_names),
                index_buffer=attributes_buffer._index_buffer_,
                mode=attributes_buffer._primitive_mode_
            ),
            texture_bindings=tuple(texture_bindings),
            uniform_block_bindings=tuple(uniform_block_bindings)
        )

    def render(
        self: Self,
        framebuffer: Framebuffer
    ) -> None:
        if (vertex_array_info := self._vertex_array_info_) is None:
            return

        uniform_block_buffer_dict = {
            uniform_block_buffer._name_: uniform_block_buffer
            for uniform_block_buffer in self._uniform_block_buffers_
        }
        uniform_buffers = tuple(
            (uniform_block_buffer_dict[name]._buffer_, binding)
            for name, binding in vertex_array_info.uniform_block_bindings
        )
        framebuffer._render_msaa(
            textures=vertex_array_info.texture_bindings,
            uniform_buffers=uniform_buffers,
            vertex_array=vertex_array_info.vertex_array
        )
