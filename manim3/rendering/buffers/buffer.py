from __future__ import annotations


#import functools
#import operator
from typing import Self

from ...constants.custom_typing import ShapeType
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject


class Buffer(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        shape: ShapeType | None = None,
        #fields: tuple[str, ...],
        #structs: dict[str, tuple[str, ...]] | None = None,
        array_lens: dict[str, int] | None = None
    ) -> None:
        super().__init__()
        #self._fields_ = fields
        #if structs is not None:
        #    self._struct_items_ = tuple(structs.items())
        if shape is not None:
            self._shape_ = shape
        if array_lens is not None:
            self._array_len_items_ = tuple(array_lens.items())

    @Lazy.variable()
    @staticmethod
    def _shape_() -> ShapeType:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _array_len_items_() -> tuple[tuple[str, int], ...]:
        return ()

    #@Lazy.property()
    #@staticmethod
    #def _size_(
    #    shape: ShapeType
    #) -> int:
    #    return functools.reduce(operator.mul, shape, initial=1)

    @Lazy.property(plural=True)
    @staticmethod
    def _macros_(
        array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[str, ...]:
        return tuple(
            f"#define {array_len_name} {array_len}"
            for array_len_name, array_len in array_len_items
            #if not re.fullmatch(r"__\w+__", array_len_name)
        )

    #@Lazy.property()
    #@staticmethod
    #def _layout_() -> type[BufferLayout]:
    #    return DenseBufferLayout

    #@Lazy.property()
    #@staticmethod
    #def _buffer_format_(
    #    fields: tuple[str, ...],
    #    struct_items: tuple[tuple[str, tuple[str, ...]], ...],
    #    array_len_items: tuple[tuple[str, int], ...],
    #    layout: type[BufferLayout]
    #) -> StructuredBufferFormat:
    #    return layout._get_structured_buffer_format(
    #        fields=fields,
    #        struct_dict=dict(struct_items),
    #        array_lens_dict=dict(array_len_items)
    #    )

        #return get_buffer_format_item(
        #    field=field,
        #    struct_dict=dict(struct_items),
        #    array_lens_dict=dict(array_len_items)
        #)

    #@Lazy.property(plural=True)
    #@staticmethod
    #def _buffer_pointer_keys_(
    #    buffer_format__pointers: tuple[tuple[tuple[str, ...], int], ...]
    #) -> tuple[str, ...]:
    #    return tuple(".".join(name_chain) for name_chain, _ in buffer_format__pointers)
