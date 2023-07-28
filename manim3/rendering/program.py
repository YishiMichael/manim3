#import itertools as it
#import re
#from dataclasses import dataclass

#import moderngl

#from ..lazy.lazy import (
#    Lazy,
#    LazyObject
#)
#from ..toplevel.toplevel import Toplevel
#from ..utils.path import PathUtils
#from .buffer_formats.atomic_buffer_format import AtomicBufferFormat
#from .buffer_formats.buffer_format import BufferFormat


#class Program(LazyObject):
#    __slots__ = ()

#    def __init__(
#        self,
#        shader_filename: str,
#        custom_macros: tuple[str, ...],
#        array_len_items: tuple[tuple[str, int], ...],
#        varyings: tuple[str, ...]
#    ) -> None:
#        super().__init__()
#        self._shader_filename_ = shader_filename
#        self._custom_macros_ = custom_macros
#        self._array_len_items_ = array_len_items
#        self._varyings_ = varyings

#    @Lazy.variable_hashable
#    @classmethod
#    def _shader_filename_(cls) -> str:
#        return ""

#    @Lazy.variable_hashable
#    @classmethod
#    def _custom_macros_(cls) -> tuple[str, ...]:
#        return ()

#    @Lazy.variable_hashable
#    @classmethod
#    def _array_len_items_(cls) -> tuple[tuple[str, int], ...]:
#        return ()

#    @Lazy.variable_hashable
#    @classmethod
#    def _varyings_(cls) -> tuple[str, ...]:
#        return ()

#    @Lazy.property_external
#    @classmethod
#    def _info_(
#        cls,
#        shader_filename: str,
#        custom_macros: tuple[str, ...],
#        array_len_items: tuple[tuple[str, int], ...],
#        varyings: tuple[str, ...]
#    ) -> ProgramInfo:

#        

#    @_info_.finalizer
#    @classmethod
#    def _info_finalizer(
#        cls,
#        info: ProgramInfo
#    ) -> None:
#        info.program.release()

#    #def _get_vertex_array(
#    #    self,
#    #    indexed_attributes_buffer: IndexedAttributesBuffer
#    #) -> moderngl.VertexArray | None:

#    #    

#    #def _get_texture_bindings(
#    #    self,
#    #    texture_buffers: list[TextureBuffer]
#    #) -> tuple[tuple[moderngl.Texture, int], ...]:
#    #    #program = self._info_.program
#    #    #texture_binding_offset_dict = self._info_.texture_binding_offset_dict
#    #    #texture_bindings: list[tuple[moderngl.Texture, int]] = []
#    #    #for texture_buffer in texture_buffers:
#    #    #    texture_buffer_format = texture_buffer._buffer_format_
#    #    #    name = texture_buffer_format._name_
#    #    #    if (texture := program.get(name, None)) is None:
#    #    #        continue

#    #    #for texture_buffer_format in self._texture_buffer_formats_:
#    #    #    if texture_buffer_format._is_empty_:
#    #    #        continue
#    #    #    name = texture_buffer_format._name_
#    #    #    if (binding_offset := texture_binding_offset_dict.get(name)) is None:
#    #    #        continue
#    #    #    texture_array = texture_array_dict[name]
#    #    #    assert texture_buffer_format._shape_ == texture_array.shape
#    #    #    texture_bindings.extend(
#    #    #        (texture, binding)
#    #    #        for binding, texture in enumerate(texture_array.flatten(), start=binding_offset)
#    #    #    )
#    #    #return tuple(texture_bindings)

#    #    uniform_info_dict = self._info_.uniform_info_dict
#    #    return tuple(
#    #        (texture, binding)
#    #        for texture_buffer in texture_buffers
#    #        if (uniform_info := uniform_info_dict.get(texture_buffer._buffer_format_._name_)) is not None
#    #        and uniform_info.verify_buffer_format(texture_buffer._buffer_format_)
#    #        for binding, texture in enumerate(texture_buffer._texture_array_.flatten(), start=uniform_info.binding)
#    #    )

#    #def _get_uniform_block_bindings(
#    #    self,
#    #    uniform_block_buffers: list[UniformBlockBuffer]
#    #) -> tuple[tuple[moderngl.Buffer, int], ...]:
#    #    #program = self._info_.program
#    #    #uniform_block_binding_dict = self._info_.uniform_block_binding_dict
#    #    #uniform_block_bindings: list[tuple[moderngl.Buffer, int]] = []
#    #    #uniform_block_info_dict = self._info_.uniform_block_info_dict
#    #    #for uniform_block_buffer in uniform_block_buffers:
#    #    #    #uniform_block_buffer_format = uniform_block_buffer._buffer_format_
#    #    #    #name = uniform_block_buffer_format._name_
#    #    #    if (uniform_block_info := uniform_block_info_dict.get(uniform_block_buffer._buffer_format_._name_)) is None:
#    #    #        continue
#    #    #    uniform_block_info.verify_buffer_format(uniform_block_buffer._buffer_format_)
#    #    #    #assert isinstance(uniform_block, moderngl.UniformBlock)
#    #    #    #assert not uniform_block_buffer_format._is_empty_
#    #    #    #assert uniform_block.size == uniform_block_buffer_format._nbytes_
#    #    #    uniform_block_bindings.append(
#    #    #        (uniform_block_buffer._buffer_, uniform_block_info.binding)
#    #    #    )
#    #    #return tuple(uniform_block_bindings)

#    #    uniform_block_info_dict = self._info_.uniform_block_info_dict
#    #    return tuple(
#    #        (uniform_block_buffer._buffer_, uniform_block_info.binding)
#    #        for uniform_block_buffer in uniform_block_buffers
#    #        if (uniform_block_info := uniform_block_info_dict.get(uniform_block_buffer._buffer_format_._name_)) is not None
#    #        and uniform_block_info.verify_buffer_format(uniform_block_buffer._buffer_format_)
#    #    )
