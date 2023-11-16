from __future__ import annotations


import itertools
import pathlib
import re
from typing import (
    Self,
    TypeGuard
)

import attrs
import moderngl
import numpy as np

from ..lazy.lazy import Lazy
from ..lazy.lazy_object import LazyObject
from ..toplevel.toplevel import Toplevel
from ..utils.path_utils import PathUtils
from .buffer_formats.atomic_buffer_format import AtomicBufferFormat
from .buffer_formats.buffer_format import BufferFormat
from .buffer_formats.structured_buffer_format import StructuredBufferFormat
from .buffers.attributes_buffer import AttributesBuffer
from .buffers.texture_buffer import TextureBuffer
from .buffers.transform_feedback_buffer import TransformFeedbackBuffer
from .buffers.uniform_block_buffer import UniformBlockBuffer
from .framebuffers.framebuffer import Framebuffer
from .indexed_attributes_buffer import IndexedAttributesBuffer
from .mgl_enums import PrimitiveMode


@attrs.frozen(kw_only=True)
class ProgramAttributeInfo:
    array_length: int
    dimension: int
    shape: str

    def verify_buffer_format(
        self: Self,
        buffer_format: BufferFormat
    ) -> TypeGuard[AtomicBufferFormat]:
        assert isinstance(buffer_format, AtomicBufferFormat)
        assert not buffer_format._is_empty_
        assert buffer_format._size_ == self.array_length
        assert buffer_format._col_len_ * buffer_format._row_len_ == self.dimension
        assert buffer_format._base_char_.replace("u", "I") == self.shape
        return True


@attrs.frozen(kw_only=True)
class ProgramUniformInfo:
    array_length: int
    binding: int

    def verify_buffer_format(
        self: Self,
        buffer_format: BufferFormat
    ) -> TypeGuard[AtomicBufferFormat]:
        assert isinstance(buffer_format, AtomicBufferFormat)
        assert not buffer_format._is_empty_
        match buffer_format._shape_:
            case ():
                array_length = 1
            case (array_length,):
                pass
            case _:
                raise ValueError
        assert array_length == self.array_length
        return True


@attrs.frozen(kw_only=True)
class ProgramUniformBlockInfo:
    size: int
    binding: int

    def verify_buffer_format(
        self: Self,
        buffer_format: BufferFormat
    ) -> TypeGuard[StructuredBufferFormat]:
        assert isinstance(buffer_format, StructuredBufferFormat)
        assert not buffer_format._is_empty_
        assert buffer_format._nbytes_ == self.size
        return True


@attrs.frozen(kw_only=True)
class ProgramInfo:
    program: moderngl.Program
    attribute_info_dict: dict[str, ProgramAttributeInfo]
    uniform_info_dict: dict[str, ProgramUniformInfo]
    uniform_block_info_dict: dict[str, ProgramUniformBlockInfo]


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        shader_path: pathlib.Path,
        custom_macros: tuple[str, ...] | None = None,
        texture_buffers: tuple[TextureBuffer, ...] | None = None,
        uniform_block_buffers: tuple[UniformBlockBuffer, ...] | None = None,
        indexed_attributes_buffer: IndexedAttributesBuffer | None = None,
        transform_feedback_buffer: TransformFeedbackBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_path_ = shader_path
        if custom_macros is not None:
            self._custom_macros_ = custom_macros
        if texture_buffers is not None:
            self._texture_buffers_ = texture_buffers
        if uniform_block_buffers is not None:
            self._uniform_block_buffers_ = uniform_block_buffers
        if indexed_attributes_buffer is not None:
            self._indexed_attributes_buffer_ = indexed_attributes_buffer
        if transform_feedback_buffer is not None:
            self._transform_feedback_buffer_ = transform_feedback_buffer

    @Lazy.variable()
    @staticmethod
    def _shader_path_() -> pathlib.Path:
        return NotImplemented

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
    def _indexed_attributes_buffer_() -> IndexedAttributesBuffer:
        return IndexedAttributesBuffer(
            attributes_buffer=AttributesBuffer(
                fields=(),
                num_vertex=0,
                data={}
            ),
            mode=PrimitiveMode.POINTS
        )

    @Lazy.variable()
    @staticmethod
    def _transform_feedback_buffer_() -> TransformFeedbackBuffer:
        return TransformFeedbackBuffer(
            fields=(),
            num_vertex=0
        )

    @Lazy.property(plural=True)
    @staticmethod
    def _array_len_items_(
        texture_buffers__array_len_items: tuple[tuple[tuple[str, int], ...], ...],
        uniform_block_buffers__array_len_items: tuple[tuple[tuple[str, int], ...], ...],
        indexed_attributes_buffer__attributes_buffer__array_len_items: tuple[tuple[str, int], ...],
        transform_feedback_buffer__array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        return tuple(
            (array_len_name, array_len)
            for array_len_name, array_len in itertools.chain(
                itertools.chain.from_iterable(texture_buffers__array_len_items),
                itertools.chain.from_iterable(uniform_block_buffers__array_len_items),
                indexed_attributes_buffer__attributes_buffer__array_len_items,
                transform_feedback_buffer__array_len_items
            )
            if not re.fullmatch(r"__\w+__", array_len_name)
        )

    @Lazy.property()
    @staticmethod
    def _program_info_(
        shader_path: pathlib.Path,
        custom_macros: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...],
        transform_feedback_buffer__buffer_pointer_keys: tuple[str, ...]
    ) -> ProgramInfo:

        def read_shader_with_includes_replaced(
            shader_path: pathlib.Path
        ) -> str:
            shader_text = shader_path.read_text(encoding="utf-8")
            return re.sub(
                r"#include \"(.+?)\"",
                lambda match: read_shader_with_includes_replaced(
                    PathUtils.shaders_dir.joinpath(match.group(1))
                ),
                shader_text
            )

        shader_text = read_shader_with_includes_replaced(shader_path)
        shaders = {
            shader_type: "\n".join((
                f"#version {Toplevel.context.version_code} core",
                "\n",
                f"#define {shader_type}",
                *custom_macros,
                *(
                    f"#define {array_len_name} {array_len}"
                    for array_len_name, array_len in array_len_items
                ),
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
        program = Toplevel.context.program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER"),
            varyings=transform_feedback_buffer__buffer_pointer_keys
        )

        attribute_info_dict: dict[str, ProgramAttributeInfo] = {}
        uniform_info_dict: dict[str, ProgramUniformInfo] = {}
        uniform_block_info_dict: dict[str, ProgramUniformBlockInfo] = {}
        texture_binding = 1
        uniform_block_binding = 0
        for name in program:
            assert re.fullmatch(r"\w+", name)

            match program[name]:
                case moderngl.Attribute() as attribute:
                    attribute_info_dict[name] = ProgramAttributeInfo(
                        array_length=attribute.array_length,
                        dimension=attribute.dimension,
                        shape=attribute.shape
                    )

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

        return ProgramInfo(
            program=program,
            attribute_info_dict=attribute_info_dict,
            uniform_info_dict=uniform_info_dict,
            uniform_block_info_dict=uniform_block_info_dict
        )

    @Lazy.property()
    @staticmethod
    def _vertex_array_(
        indexed_attributes_buffer: IndexedAttributesBuffer,
        program_info: ProgramInfo
    ) -> moderngl.VertexArray | None:
        attributes_buffer = indexed_attributes_buffer._attributes_buffer_
        index_buffer = indexed_attributes_buffer._index_buffer_
        mode = indexed_attributes_buffer._mode_
        assert isinstance(attributes_buffer_format := attributes_buffer._buffer_format_, StructuredBufferFormat)
        use_index_buffer = not index_buffer._omitted

        if attributes_buffer_format._is_empty_ or use_index_buffer and index_buffer._buffer_format_._is_empty_:
            return None

        attribute_items = tuple(
            (child, offset)
            for child, offset in zip(attributes_buffer_format._children_, attributes_buffer_format._offsets_, strict=True)
            if (attribute_info := program_info.attribute_info_dict.get(child._name_)) is not None
            and attribute_info.verify_buffer_format(child)
        )

        components: list[str] = []
        current_offset = 0
        for child, offset in attribute_items:
            if (padding := offset - current_offset):
                components.append(f"{padding}x")
            components.append(child._format_str_)
            current_offset = offset + child._nbytes_
        if (padding := attributes_buffer_format._itemsize_ - current_offset):
            components.append(f"{padding}x")
        components.append("/v")

        return Toplevel.context.vertex_array(
            program=program_info.program,
            attributes_buffer=attributes_buffer._buffer_,
            buffer_format_str=" ".join(components),
            attribute_names=tuple(child._name_ for child, _ in attribute_items),
            index_buffer=index_buffer._buffer_ if use_index_buffer else None,
            mode=mode
        )

    @Lazy.property(plural=True)
    @staticmethod
    def _texture_bindings_(
        texture_buffers: tuple[TextureBuffer, ...],
        program_info: ProgramInfo
    ) -> tuple[tuple[moderngl.Texture, int], ...]:
        return tuple(
            (texture, binding)
            for texture_buffer in texture_buffers
            if (uniform_info := program_info.uniform_info_dict.get(texture_buffer._buffer_format_._name_)) is not None
            and uniform_info.verify_buffer_format(texture_buffer._buffer_format_)
            for binding, texture in enumerate(texture_buffer._texture_array_.flatten(), start=uniform_info.binding)
        )

    @Lazy.property(plural=True)
    @staticmethod
    def _uniform_block_bindings_(
        uniform_block_buffers: tuple[UniformBlockBuffer, ...],
        program_info: ProgramInfo
    ) -> tuple[tuple[moderngl.Buffer, int], ...]:
        return tuple(
            (uniform_block_buffer._buffer_, uniform_block_info.binding)
            for uniform_block_buffer in uniform_block_buffers
            if (uniform_block_info := program_info.uniform_block_info_dict.get(uniform_block_buffer._buffer_format_._name_)) is not None
            and uniform_block_info.verify_buffer_format(uniform_block_buffer._buffer_format_)
        )

    def render(
        self: Self,
        framebuffer: Framebuffer
    ) -> None:
        if (vertex_array := self._vertex_array_) is None:
            return

        with Toplevel.context.scope(
            framebuffer=framebuffer._framebuffer_,
            textures=self._texture_bindings_,
            uniform_buffers=self._uniform_block_bindings_
        ):
            Toplevel.context.set_state(framebuffer._context_state_)
            vertex_array.render()

    def transform(
        self: Self
    ) -> dict[str, np.ndarray]:
        transform_feedback_buffer = self._transform_feedback_buffer_
        with transform_feedback_buffer.buffer() as buffer:
            if (vertex_array := self._vertex_array_) is not None:
                with Toplevel.context.scope(
                    uniform_buffers=self._uniform_block_bindings_
                ):
                    vertex_array.transform(buffer=buffer)
            data_dict = transform_feedback_buffer.read(buffer)
        return data_dict
