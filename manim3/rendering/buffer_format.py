#from __future__ import annotations


#import functools
#import itertools
#import operator
#from typing import Self

#import numpy as np

#from ..constants.custom_typing import ShapeType
#from ..lazy.lazy import Lazy
#from ..lazy.lazy_object import LazyObject


#class BufferFormat(LazyObject):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        *,
#        #name: str,
#        itemsize: int,
#        base_alignment: int
#        #shape: ShapeType
#    ) -> None:
#        super().__init__()
#        #self._name_ = name
#        self._itemsize_ = itemsize
#        self._base_alignment_ = base_alignment
#        #self._shape_ = shape

#    #@Lazy.variable()
#    #@staticmethod
#    #def _shape_() -> ShapeType:
#    #    return ()

#    @Lazy.variable()
#    @staticmethod
#    def _itemsize_() -> int:
#        return 0

#    @Lazy.variable()
#    @staticmethod
#    def _base_alignment_() -> int:
#        return 1

#    #@Lazy.property()
#    #@staticmethod
#    #def _size_(
#    #    shape: ShapeType
#    #) -> int:
#    #    return int(np.prod(shape, dtype=np.int32))

#    #@Lazy.property()
#    #@staticmethod
#    #def _nbytes_(
#    #    itemsize: int,
#    #    size: int
#    #) -> int:
#    #    return itemsize * size

#    #@Lazy.property()
#    #@staticmethod
#    #def _is_empty_(
#    #    size: int
#    #) -> bool:
#    #    return not size


#class AtomicBufferFormat(BufferFormat):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        *,
#        #name: str,
#        itemsize: int,
#        base_alignment: int,
#        #shape: tuple[int, ...],
#        base_char: str,
#        base_itemsize: int,
#        base_ndim: int,
#        row_len: int,
#        col_len: int,
#        col_padding: int
#    ) -> None:
#        super().__init__(
#            #name=name,
#            itemsize=itemsize,
#            base_alignment=base_alignment
#            #shape=shape
#        )
#        self._base_char_ = base_char
#        self._base_itemsize_ = base_itemsize
#        self._base_ndim_ = base_ndim
#        self._row_len_ = row_len
#        self._col_len_ = col_len
#        self._col_padding_ = col_padding


#class StructuredBufferFormat(BufferFormat):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        *,
#        #name: str,
#        itemsize: int,
#        base_alignment: int,
#        #shape: tuple[int, ...],
#        #children: tuple[BufferFormat, ...],
#        #names: tuple[str, ...],
#        #shapes: tuple[ShapeType, ...],
#        #offsets: tuple[int, ...]
#        #children_items: tuple[tuple[BufferFormat, str, ShapeType, int, int], ...]
#        buffer_fields: tuple[BufferField, ...]
#    ) -> None:
#        super().__init__(
#            #name=name,
#            itemsize=itemsize,
#            base_alignment=base_alignment
#            #shape=shape
#        )
#        self._buffer_fields_ = buffer_fields
#        #self._children_ = children
#        #self._names_ = names
#        #self._shapes_ = shapes
#        #self._offsets_ = offsets
#        #self._itemsize_ = itemsize
#        #self._base_alignment_ = base_alignment

#    @Lazy.variable(plural=True)
#    @staticmethod
#    def _buffer_fields_() -> tuple[BufferField, ...]:
#        return ()

#    #@Lazy.property(plural=True)
#    #@staticmethod
#    #def _children_(
#    #    children_items: tuple[tuple[BufferFormat, str, ShapeType, int, int], ...]
#    #) -> tuple[BufferFormat, ...]:
#    #    return ()

#    #@Lazy.variable(plural=True)
#    #@staticmethod
#    #def _children_() -> tuple[BufferFormat, ...]:
#    #    return ()

#    #@Lazy.variable(plural=True)
#    #@staticmethod
#    #def _names_() -> tuple[str, ...]:
#    #    return ()

#    #@Lazy.variable(plural=True)
#    #@staticmethod
#    #def _shapes_() -> tuple[ShapeType, ...]:
#    #    return ()

#    #@Lazy.variable(plural=True)
#    #@staticmethod
#    #def _offsets_() -> tuple[int, ...]:
#    #    return ()

#    @Lazy.property()
#    @staticmethod
#    def _dtype_(
#        names: tuple[str, ...],
#        buffer_fields: tuple[BufferField, ...],
#        #shapes: tuple[ShapeType, ...],
#        #offsets: tuple[int, ...],
#        itemsize: int
#    ) -> np.dtype:
#        return np.dtype({
#            "names": names,
#            "formats": tuple(
#                (buffer_field._buffer_format_._dtype_, buffer_field._shape_)
#                for buffer_field in buffer_fields
#            ),
#            "offsets": tuple(
#                buffer_field._offset_
#                for buffer_field in buffer_fields
#            ),
#            "itemsize": itemsize
#        })

#    @Lazy.property(plural=True)
#    @staticmethod
#    def _pointers_(
#        buffer_fields: tuple[BufferField, ...]
#        #names: tuple[str, ...],
#        #children__pointers: tuple[tuple[tuple[tuple[str, ...], int], ...], ...]
#    ) -> tuple[tuple[tuple[str, ...], int], ...]:
#        return tuple(
#            ((buffer_field._name_,) + name_chain, base_ndim)
#            for buffer_field in buffer_fields
#            for name_chain, base_ndim in buffer_field._buffer_format_._pointers_
#        )

#    @Lazy.property()
#    @staticmethod
#    def _format_str_(
#        buffer_fields__format_str: tuple[str, ...]
#        #size: int
#    ) -> str:
#        return " ".join(buffer_fields__format_str)
#        #components: list[str] = []
#        #current_offset = 0
#        #for buffer_field in buffer_fields:
#        #    if (padding := buffer_field._offset_ - current_offset):
#        #        components.append(f"{padding}x")
#        #    components.extend(itertools.repeat(child._format_str_, ))
#        #    current_offset = offset + child._nbytes_
#        #if (padding := attributes_buffer_itemsize - current_offset):
#        #    components.append(f"{padding}x")


#class BufferField(LazyObject):
#    __slots__ = ()

#    def __init__(
#        self: Self,
#        buffer_format: BufferFormat,
#        name: str,
#        shape: ShapeType
#        #offset: int,
#        #padding: int
#    ) -> None:
#        super().__init__()
#        self._buffer_format_ = buffer_format
#        self._name_ = name
#        self._shape_ = shape
#        #self._offset_ = offset
#        #self._padding_ = padding

#    @Lazy.variable()
#    @staticmethod
#    def _buffer_format_() -> BufferFormat:
#        return NotImplemented

#    @Lazy.variable()
#    @staticmethod
#    def _name_() -> str:
#        return ""

#    @Lazy.variable()
#    @staticmethod
#    def _shape_() -> ShapeType:
#        return ()

#    #@Lazy.variable()
#    #@staticmethod
#    #def _offset_() -> int:
#    #    return 0

#    #@Lazy.variable()
#    #@staticmethod
#    #def _padding_() -> int:
#    #    return 0

#    @Lazy.property()
#    @staticmethod
#    def _size_(
#        shape: ShapeType
#    ) -> int:
#        return functools.reduce(operator.mul, shape, initial=1)

#    #@Lazy.property()
#    #@staticmethod
#    #def _format_str_(
#    #    buffer_format__format_str: str,
#    #    size: int,
#    #    padding: int
#    #    #col_len: int,
#    #    #base_char: str,
#    #    #base_itemsize: int,
#    #    #col_padding: int,
#    #    #row_len: int
#    #    #size: int
#    #) -> str:
#    #    field_format = " ".join(itertools.repeat(buffer_format__format_str, size))
#    #    if padding:
#    #        field_format = f"{field_format} {padding}x"
#    #    return field_format
