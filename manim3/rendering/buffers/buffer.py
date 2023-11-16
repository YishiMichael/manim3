from __future__ import annotations


import re
from typing import Self

from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject
from ..buffer_formats.buffer_format import BufferFormat
from ..buffer_layouts.buffer_layout import BufferLayout
from ..buffer_layouts.dense_buffer_layout import DenseBufferLayout


class Buffer(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        field: str,
        child_structs: dict[str, tuple[str, ...]] | None,
        array_lens: dict[str, int] | None
    ) -> None:
        super().__init__()
        self._field_ = field
        if child_structs is not None:
            self._child_struct_items_ = tuple(child_structs.items())
        if array_lens is not None:
            self._array_len_items_ = tuple(array_lens.items())

    @Lazy.variable()
    @staticmethod
    def _field_() -> str:
        return ""

    @Lazy.variable(plural=True)
    @staticmethod
    def _child_struct_items_() -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _array_len_items_() -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _layout_() -> type[BufferLayout]:
        return DenseBufferLayout

    @Lazy.property()
    @staticmethod
    def _buffer_format_(
        field: str,
        child_struct_items: tuple[tuple[str, tuple[str, ...]], ...],
        array_len_items: tuple[tuple[str, int], ...],
        layout: type[BufferLayout]
    ) -> BufferFormat:

        def parse_field_str(
            field_str: str,
            array_lens_dict: dict[str, int]
        ) -> tuple[str, str, tuple[int, ...]]:
            pattern = re.compile(r"""
                (?P<dtype_str>\w+?)
                \s
                (?P<name>\w+?)
                (?P<shape>(\[\w+?\])*)
            """, flags=re.VERBOSE)
            match = pattern.fullmatch(field_str)
            assert match is not None
            dtype_str = match.group("dtype_str")
            name = match.group("name")
            shape = tuple(
                int(s) if re.match(r"^\d+$", s := index_match.group(1)) is not None else array_lens_dict[s]
                for index_match in re.finditer(r"\[(\w+?)\]", match.group("shape"))
            )
            return (dtype_str, name, shape)

        def get_buffer_format(
            field: str,
            child_structs_dict: dict[str, tuple[str, ...]],
            array_lens_dict: dict[str, int]
        ) -> BufferFormat:
            dtype_str, name, shape = parse_field_str(field, array_lens_dict)
            if (child_struct_fields := child_structs_dict.get(dtype_str)) is None:
                return layout.get_atomic_buffer_format(
                    name=name,
                    shape=shape,
                    gl_dtype_str=dtype_str
                )
            return layout.get_structured_buffer_format(
                name=name,
                shape=shape,
                children=tuple(
                    get_buffer_format(
                        child_struct_field,
                        child_structs_dict,
                        array_lens_dict
                    )
                    for child_struct_field in child_struct_fields
                )
            )

        return get_buffer_format(
            field,
            dict(child_struct_items),
            dict(array_len_items)
        )

    @Lazy.property(plural=True)
    @staticmethod
    def _buffer_pointer_keys_(
        buffer_format__pointers: tuple[tuple[tuple[str, ...], int], ...]
    ) -> tuple[str, ...]:
        return tuple(".".join(name_chain) for name_chain, _ in buffer_format__pointers)
