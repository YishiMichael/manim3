from __future__ import annotations


import itertools
import pathlib
import re
from typing import Self

import attrs
import moderngl
#import numpy as np

from ..lazy.lazy import Lazy
from ..lazy.lazy_object import LazyObject
from ..toplevel.toplevel import Toplevel
from ..utils.path_utils import PathUtils
#from .buffer_formats.atomic_buffer_format import AtomicBufferFormat
#from .buffer_formats.buffer_format import BufferFormat
#from .buffer_formats.structured_buffer_format import StructuredBufferFormat
from .buffers.attributes_buffer import AttributesBuffer
from .buffers.texture_buffer import TextureBuffer
#from .buffers.transform_feedback_buffer import TransformFeedbackBuffer
from .buffers.uniform_block_buffer import UniformBlockBuffer
from .framebuffers.framebuffer import Framebuffer
#from .indexed_attributes_buffer import IndexedAttributesBuffer
from .field import (
    AtomicField,
    #Field,
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
        #assert isinstance(buffer_format, AtomicBufferFormat)
        #match texture_buffer._shape_:
        #    case ():
        #        array_length = 1
        #    case (array_length,):
        #        assert array_length
        #    case _:
        #        raise AssertionError
        assert len(texture_buffer._textures_) == self.array_length
        #return True


@attrs.frozen(kw_only=True)
class ProgramUniformBlockInfo:
    size: int
    binding: int

    def verify_structured_field(
        self: Self,
        field: StructuredField
    ) -> None:
        #assert isinstance(buffer_format, StructuredBufferFormat)
        #assert not field._is_empty_
        assert field._itemsize_ == self.size
        #return True


@attrs.frozen(kw_only=True)
class ProgramAttributeInfo:
    array_length: int
    dimension: int
    shape: str

    def verify_atomic_field(
        self: Self,
        field: AtomicField
    ) -> None:
        #assert isinstance(field, AtomicBufferFormat)
        #assert field._nbytes_
        assert field._size_ == self.array_length
        assert field._col_len_ * field._row_len_ == self.dimension
        assert field._base_char_.replace("u", "I") == self.shape
        #return True


@attrs.frozen(kw_only=True)
class VertexArrayInfo:
    vertex_array: moderngl.VertexArray
    texture_bindings: tuple[tuple[moderngl.Texture, int], ...]
    uniform_block_bindings: tuple[tuple[str, int], ...]


#@attrs.frozen(kw_only=True)
#class ProgramInfo:
#    program: moderngl.Program
#    uniform_info_dict: dict[str, ProgramUniformInfo]
#    uniform_block_info_dict: dict[str, ProgramUniformBlockInfo]
#    attribute_info_dict: dict[str, ProgramAttributeInfo]


#@attrs.frozen(kw_only=True)
#class ModernglBuffers:
#    mgl_uniform_block_buffers: tuple[tuple[moderngl.Buffer, str], ...]
#    mgl_attributes_buffer: moderngl.Buffer
#    mgl_index_buffer: moderngl.Buffer | None
#    mgl_vertex_array: moderngl.VertexArray
#    mgl_texture_bindings: tuple[tuple[moderngl.Texture, int], ...]
#    mgl_uniform_block_bindings: tuple[tuple[moderngl.Buffer, int], ...]
#    program: moderngl.Program
#    attributes_buffer_format_str: str
#    attribute_names: tuple[str, ...]
#    mode: PrimitiveMode

#    def render(
#        self: Self,
#        attributes_buffer_bytes: bytes,
#        index_buffer_bytes: bytes | None,
#        uniform_block_buffer_bytes_dict: dict[str, bytes],
#        framebuffer: Framebuffer
#    ) -> None:

#        def write_buffer(
#            buffer: moderngl.Buffer,
#            data_bytes: bytes
#        ) -> None:
#            if buffer.dynamic:
#                buffer.orphan(len(data_bytes))
#            else:
#                buffer.orphan()
#            buffer.write(data_bytes)

#        #mgl_attributes_buffer = Toplevel.context.buffer(data=attributes_buffer_bytes)
#        write_buffer(
#            self.mgl_attributes_buffer,
#            attributes_buffer_bytes
#            #orphan=True
#        )
#        #if index_buffer_bytes is None:
#        #    mgl_index_buffer = None
#        #else:
#        #    mgl_index_buffer = Toplevel.context.buffer(data=index_buffer_bytes)
#        if index_buffer_bytes is not None:
#            assert self.mgl_index_buffer is not None
#            write_buffer(
#                self.mgl_index_buffer,
#                index_buffer_bytes
#                #orphan=True
#            )
#        else:
#            assert self.mgl_index_buffer is None

#        #if attributes_buffer._use_index_buffer_:
#        #    assert mgl_index_buffer is not None
#        #    if not attributes_buffer._index_bytes_:
#        #        return
#        #    write_buffer(
#        #        mgl_index_buffer,
#        #        attributes_buffer._index_bytes_
#        #        #orphan=True
#        #    )

#        #mgl_uniform_block_buffer_items = tuple(
#        #    Toplevel.context.buffer(data=uniform_block_buffer_bytes_dict[name])
#        #    for _, name in self.mgl_uniform_block_buffers
#        #)
#        #mgl_uniform_block_bindings = tuple(
#        #    (mgl_buffer, binding)
#        #    for mgl_buffer, (_, binding) in zip(mgl_uniform_block_buffer_items, self.mgl_uniform_block_bindings, strict=True)
#        #)
#        for mgl_uniform_block_buffer, name in self.mgl_uniform_block_buffers:
#            write_buffer(
#                mgl_uniform_block_buffer,
#                uniform_block_buffer_bytes_dict[name]
#                #orphan=False
#            )
#        mgl_vertex_array = Toplevel.context.vertex_array(
#            program=self.program,
#            attributes_buffer=self.mgl_attributes_buffer,
#            attributes_buffer_format_str=self.attributes_buffer_format_str,
#            attribute_names=self.attribute_names,
#            index_buffer=self.mgl_index_buffer,
#            mode=self.mode
#        )

#        with Toplevel.context.scope(
#            framebuffer=framebuffer._framebuffer_,
#            textures=self.mgl_texture_bindings,
#            uniform_buffers=self.mgl_uniform_block_bindings
#        ):
#            Toplevel.context.set_state(framebuffer._context_state_)
#            mgl_vertex_array.render()


#    def fetch_moderngl_buffers(
#        self: Self
#    ) -> ModernglBuffers:
#        mgl_attributes_buffer = Toplevel.context.buffer(reserve=1, dynamic=True)
#        mgl_index_buffer = Toplevel.context.buffer(reserve=1, dynamic=True) if self._use_index_buffer else None
#        mgl_uniform_block_buffer_items = tuple(
#            (Toplevel.context.buffer(reserve=max(itemsize, 1)), name, binding)
#            for name, itemsize, binding in self._uniform_block_items
#        )
#        #mgl_uniform_block_buffers = {
#        #    name: Toplevel.context.buffer(reserve=max(itemsize, 1))
#        #    for name, itemsize, _ in self._uniform_block_items
#        #}
#        #print(self._attribute_names)
#        #print(self._attributes_buffer_format_str)
#        mgl_vertex_array = 
#        return ModernglBuffers(
#            mgl_uniform_block_buffers=tuple(
#                (mgl_uniform_block_buffer, name)
#                for mgl_uniform_block_buffer, name, _ in mgl_uniform_block_buffer_items
#            ),
#            mgl_attributes_buffer=mgl_attributes_buffer,
#            mgl_index_buffer=mgl_index_buffer,
#            mgl_vertex_array=mgl_vertex_array,
#            mgl_texture_bindings=self._texture_bindings,
#            mgl_uniform_block_bindings=tuple(
#                (mgl_uniform_block_buffer, binding)
#                for mgl_uniform_block_buffer, _, binding in mgl_uniform_block_buffer_items
#            ),
#            program=self._program,
#            attributes_buffer_format_str=self._attributes_buffer_format_str,
#            attribute_names=self._attribute_names,
#            mode=self._primitive_mode
#        )


class VertexArray(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        shader_path: pathlib.Path,
        custom_macros: tuple[str, ...] = (),
        texture_buffers: tuple[TextureBuffer, ...] = (),
        uniform_block_buffers: tuple[UniformBlockBuffer, ...] = (),
        attributes_buffer: AttributesBuffer
        #transform_feedback_buffer: TransformFeedbackBuffer | None = None
    ) -> None:
        super().__init__()
        self._shader_path_ = shader_path
        self._custom_macros_ = custom_macros
        self._texture_buffers_ = texture_buffers
        self._uniform_block_buffers_ = uniform_block_buffers
        self._attributes_buffer_ = attributes_buffer
        #if transform_feedback_buffer is not None:
        #    self._transform_feedback_buffer_ = transform_feedback_buffer

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
    def _attributes_buffer_() -> AttributesBuffer:
        return NotImplemented

    #@Lazy.variable()
    #@staticmethod
    #def _transform_feedback_buffer_() -> TransformFeedbackBuffer:
    #    return TransformFeedbackBuffer(
    #        fields=(),
    #        num_vertex=0
    #    )

    @Lazy.property(plural=True)
    @staticmethod
    def _macros_(
        custom_macros: tuple[str, ...],
        texture_buffers__macros: tuple[tuple[str, ...], ...],
        uniform_block_buffers__macros: tuple[tuple[str, ...], ...],
        attributes_buffer__macros: tuple[str, ...]
        #transform_feedback_buffer__macros: tuple[str, ...]
    ) -> tuple[str, ...]:
        return tuple(itertools.chain(
            custom_macros,
            itertools.chain.from_iterable(texture_buffers__macros),
            itertools.chain.from_iterable(uniform_block_buffers__macros),
            attributes_buffer__macros
            #transform_feedback_buffer__macros
        ))

    #@Lazy.property()
    #@staticmethod
    #def _attributes_buffer_format_items_and_itemsize_(
    #    attributes_buffer__buffer_format: BufferFormat
    #) -> tuple[tuple[tuple[BufferFormat, int], ...], int]:
    #    assert isinstance((buffer_format := attributes_buffer__buffer_format), StructuredBufferFormat)
    #    return tuple(zip(buffer_format._children_, buffer_format._offsets_, strict=True)), buffer_format._itemsize_

    @Lazy.property()
    @staticmethod
    def _program_(
        shader_path: pathlib.Path,
        macros: tuple[str, ...],
    ) -> moderngl.Program:
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
                *macros,
                #*(
                #    f"#define {array_len_name} {array_len}"
                #    for array_len_name, array_len in array_len_items
                #),
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
        return Toplevel.context.program(
            vertex_shader=shaders["VERTEX_SHADER"],
            fragment_shader=shaders.get("FRAGMENT_SHADER"),
            geometry_shader=shaders.get("GEOMETRY_SHADER"),
            tess_control_shader=shaders.get("TESS_CONTROL_SHADER"),
            tess_evaluation_shader=shaders.get("TESS_EVALUATION_SHADER")
            #varyings=transform_feedback_buffer__buffer_pointer_keys
        )

    @Lazy.property()
    @staticmethod
    def _vertex_array_info_(
        program: moderngl.Program,
        texture_buffers: tuple[TextureBuffer, ...],
        uniform_block_buffers__field: tuple[StructuredField, ...],
        #uniform_block_buffers__name: tuple[str, ...],
        #uniform_block_buffers__buffer_format: tuple[StructuredBufferFormat, ...],
        #attributes_buffer_format_items_and_itemsize: tuple[tuple[tuple[BufferFormat, int], ...], int],
        attributes_buffer: AttributesBuffer
        #attributes_buffer__fields: tuple[AtomicField, ...],
        #attributes_buffer__merged_field__paddings: tuple[int, ...],
        ##attributes_buffer__merged_field__itemsize: int,
        ##attributes_buffer__merged_field__format_str: str,
        #attributes_buffer__use_index_buffer: bool,
        #attributes_buffer__primitive_mode: PrimitiveMode
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
        #texture_bindings = tuple(
        #    (texture, binding)
        #    for texture_buffer in texture_buffers
        #    if texture_buffer._textures_
        #    and (uniform_info := uniform_info_dict.pop(texture_buffer._name_, None)) is not None
        #    and uniform_info.verify_texture_buffer(texture_buffer)
        #    for binding, texture in enumerate(texture_buffer._textures_, start=uniform_info.binding)
        #)

        uniform_block_bindings: list[tuple[str, int]] = []
        for field in uniform_block_buffers__field:
            if (uniform_block_info := uniform_block_info_dict.pop(field._name_, None)) is None:
                continue
            uniform_block_info.verify_structured_field(field)
            uniform_block_bindings.append((field._name_, uniform_block_info.binding))

        #uniform_block_items = tuple(
        #    (field._name_, field._nbytes_, uniform_block_info.binding)
        #    for field in uniform_block_buffers__field
        #    if field._nbytes_
        #    and (uniform_block_info := uniform_block_info_dict.pop(field._name_, None)) is not None
        #    and uniform_block_info.verify_structured_field(field)
        #)

        #attributes_buffer_format_items, attributes_buffer_itemsize = attributes_buffer_format_items_and_itemsize
        #attribute_items = tuple(
        #    (field, (
        #        (attribute_info := attribute_info_dict.pop(field._name_, None)) is not None
        #        and attribute_info.verify_atomic_field(field)
        #    ))
        #    for field in attributes_buffer__fields
        #)
        #attribute_names = tuple(field._name_ for field, _ in attribute_items)
        #print(program_info.uniform_info_dict)
        #print(texture_bindings)
        #print(program_info.uniform_block_info_dict)
        #print(uniform_block_items)
        #print(program_info.attribute_info_dict)
        #print(attribute_items)
        #print(attribute_names)

        attribute_names: list[str] = []
        format_components: list[str] = []
        #current_offset = 0
        for field, padding in zip(attributes_buffer._fields_, attributes_buffer._merged_field_._paddings_, strict=True):
            #if (padding := offset - current_offset):
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
                    #return " ".join(itertools.repeat(" ".join(row_format_components), row_len))
            if padding:
                format_components.append(f"{padding}x")

            #format_components.extend(itertools.chain.from_iterable(itertools.repeat(row_format_components, field._row_len_)))
            #current_offset = offset + field._nbytes_
        #if (padding := attributes_buffer__merged_field__itemsize - current_offset):
        #format_components.append(f"{attributes_buffer__merged_field__itemsize - current_offset}x")
        format_components.append("/v")

        #attributes_buffer_format_str = " ".join(format_components)
        #print()
        #print(attribute_items)
        #print(format_components)

        assert not uniform_info_dict
        assert not uniform_block_info_dict
        assert not attribute_info_dict

        if (
            not attributes_buffer._num_vertices_
            or not attributes_buffer._merged_field_._itemsize_
            or attributes_buffer._use_index_buffer_ and not attributes_buffer._index_bytes_
        ):
            return None
        return VertexArrayInfo(
            vertex_array=Toplevel.context.vertex_array(
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

        #self._program: moderngl.Program = program
        #self._texture_bindings: tuple[tuple[moderngl.Texture, int], ...] = tuple(texture_bindings)
        #self._uniform_block_items: tuple[tuple[str, int, int], ...] = tuple(uniform_block_items)
        #self._attributes_buffer_format_str: str = " ".join(format_components)
        #self._attribute_names: tuple[str, ...] = tuple(attribute_names)
        #self._use_index_buffer: bool = use_index_buffer
        #self._primitive_mode: PrimitiveMode = primitive_mode

    #@Lazy.property()
    #@staticmethod
    #def _vertex_array_(
    #    indexed_attributes_buffer: IndexedAttributesBuffer,
    #    program_info: ProgramInfo
    #) -> moderngl.VertexArray | None:
    #    attributes_buffer = indexed_attributes_buffer._attributes_buffer_
    #    index_buffer = indexed_attributes_buffer._index_buffer_
    #    mode = indexed_attributes_buffer._mode_
    #    assert isinstance(attributes_buffer_format := attributes_buffer._buffer_format_, StructuredBufferFormat)
    #    use_index_buffer = not index_buffer._omitted

    #    if attributes_buffer_format._is_empty_ or use_index_buffer and index_buffer._buffer_format_._is_empty_:
    #        return None

    #    attribute_items = tuple(
    #        (child, offset)
    #        for child, offset in zip(attributes_buffer_format._children_, attributes_buffer_format._offsets_, strict=True)
    #        if (attribute_info := program_info.attribute_info_dict.get(child._name_)) is not None
    #        and attribute_info.verify_buffer_format(child)
    #    )

    #    components: list[str] = []
    #    current_offset = 0
    #    for child, offset in attribute_items:
    #        if (padding := offset - current_offset):
    #            components.append(f"{padding}x")
    #        components.append(child._format_str_)
    #        current_offset = offset + child._nbytes_
    #    if (padding := attributes_buffer_format._itemsize_ - current_offset):
    #        components.append(f"{padding}x")
    #    components.append("/v")

    #    return Toplevel.context.vertex_array(
    #        program=program_info.program,
    #        attributes_buffer=attributes_buffer._buffer_,
    #        attributes_buffer_format_str=" ".join(components),
    #        attribute_names=tuple(child._name_ for child, _ in attribute_items),
    #        index_buffer=index_buffer._buffer_ if use_index_buffer else None,
    #        mode=mode
    #    )

    #@Lazy.property(plural=True)
    #@staticmethod
    #def _texture_bindings_(
    #    texture_buffers: tuple[TextureBuffer, ...],
    #    program_info: ProgramInfo
    #) -> tuple[tuple[moderngl.Texture, int], ...]:
    #    return tuple(
    #        (texture, binding)
    #        for texture_buffer in texture_buffers
    #        if (uniform_info := program_info.uniform_info_dict.get(texture_buffer._buffer_format_._name_)) is not None
    #        and uniform_info.verify_buffer_format(texture_buffer._buffer_format_)
    #        for binding, texture in enumerate(texture_buffer._texture_array_.flatten(), start=uniform_info.binding)
    #    )

    #@Lazy.property(plural=True)
    #@staticmethod
    #def _uniform_block_bindings_(
    #    uniform_block_buffers: tuple[UniformBlockBuffer, ...],
    #    program_info: ProgramInfo
    #) -> tuple[tuple[moderngl.Buffer, int], ...]:
    #    return tuple(
    #        (uniform_block_buffer._buffer_, uniform_block_info.binding)
    #        for uniform_block_buffer in uniform_block_buffers
    #        if (uniform_block_info := program_info.uniform_block_info_dict.get(uniform_block_buffer._buffer_format_._name_)) is not None
    #        and uniform_block_info.verify_buffer_format(uniform_block_buffer._buffer_format_)
    #    )

    #def fetch_moderngl_buffers(
    #    self: Self
    #) -> ModernglBuffers:
    #    return self._moderngl_buffers_factory_.fetch_moderngl_buffers()

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
        with Toplevel.context.scope(
            framebuffer=framebuffer._framebuffer_,
            textures=vertex_array_info.texture_bindings,
            uniform_buffers=tuple(
                (uniform_block_buffer_dict[name]._buffer_, binding)
                for name, binding in vertex_array_info.uniform_block_bindings
            )
        ):
            Toplevel.context.set_state(framebuffer._context_state_)
            vertex_array_info.vertex_array.render()

        #attributes_buffer = self._attributes_buffer_
        #if not attributes_buffer._num_vertices_ or not attributes_buffer._merged_field_._itemsize_:
        #    return
        #attributes_buffer_bytes = attributes_buffer._data_bytes_

        #if attributes_buffer._use_index_buffer_:
        #    index_buffer_bytes = attributes_buffer._index_bytes_
        #    if not index_buffer_bytes:
        #        return
        #else:
        #    index_buffer_bytes = None

        #moderngl_buffers.render(
        #    attributes_buffer_bytes=attributes_buffer_bytes,
        #    index_buffer_bytes=index_buffer_bytes,
        #    uniform_block_buffer_bytes_dict={
        #        uniform_block_buffer._name_: uniform_block_buffer._data_bytes_
        #        for uniform_block_buffer in self._uniform_block_buffers_
        #    },
        #    framebuffer=framebuffer
        #)

    #def transform(
    #    self: Self
    #) -> dict[str, np.ndarray]:
    #    transform_feedback_buffer = self._transform_feedback_buffer_
    #    with transform_feedback_buffer.buffer() as buffer:
    #        if (vertex_array := self._vertex_array_) is not None:
    #            with Toplevel.context.scope(
    #                uniform_buffers=self._uniform_block_bindings_
    #            ):
    #                vertex_array.transform(buffer=buffer)
    #        data_dict = transform_feedback_buffer.read(buffer)
    #    return data_dict
