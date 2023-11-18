from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.custom_typing import (
    NP_xi4,
    ShapeType
)
from ...lazy.lazy import Lazy
#from ..buffer_format import (
#    AtomicBufferFormat,
#    StructuredBufferFormat
#)
from ..mgl_enums import PrimitiveMode
#from ..std140_layout import STD140Layout
from .buffer import Buffer
from ..field import (
    AtomicField,
    Field,
    StructuredField
)


class AttributesBuffer(Buffer):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        field_declarations: tuple[str, ...],
        data_dict: dict[str, np.ndarray],
        index: NP_xi4 | None = None,
        primitive_mode: PrimitiveMode,
        num_vertices: int,
        array_lens: dict[str, int] | None = None
    ) -> None:
        super().__init__(
            #field_declarations=tuple(field_declarations_dict),
            shape=(num_vertices,),
            array_lens=array_lens
        )
        self._field_declarations_ = field_declarations
        self._data_dict_ = data_dict
        self._num_vertices_ = num_vertices
        if index is not None:
            self._index_bytes_ = index.astype(np.uint32).tobytes()
            self._use_index_buffer_ = True
        self._primitive_mode_ = primitive_mode

    @Lazy.variable(plural=True)
    @staticmethod
    def _field_declarations_() -> tuple[str, ...]:
        return ()

    @Lazy.variable()
    @staticmethod
    def _data_dict_() -> dict[str, np.ndarray]:
        return {}

    @Lazy.variable()
    @staticmethod
    def _num_vertices_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _index_bytes_() -> bytes:
        return b""

    @Lazy.variable()
    @staticmethod
    def _use_index_buffer_() -> bool:
        return False

    @Lazy.variable()
    @staticmethod
    def _primitive_mode_() -> PrimitiveMode:
        return NotImplemented

    @Lazy.property(plural=True)
    @staticmethod
    def _fields_(
        field_declarations: tuple[str, ...],
        array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[AtomicField, ...]:
        return tuple(
            Field.parse_atomic_field(
                field_declaration=field_declaration,
                array_lens_dict=dict(array_len_items)
            )
            for field_declaration in field_declarations
        )

    @Lazy.property()
    @staticmethod
    def _merged_field_(
        fields: tuple[AtomicField, ...]
    ) -> StructuredField:
        return StructuredField(
            name="",
            shape=(),
            fields=fields
        )

    @Lazy.property()
    @staticmethod
    def _data_bytes_(
        merged_field: StructuredField,
        shape: ShapeType,
        data_dict: dict[str, np.ndarray]
    ) -> bytes:
        return merged_field.write(shape, data_dict)

    #@Lazy.property()
    #@staticmethod
    #def _layout_() -> type[BufferLayout]:
    #    # Let's keep using std140 layout, hopefully giving a faster processing speed.
    #    return STD140Layout

    #@Lazy.property()
    #@staticmethod
    #def _buffer_format_str_(
    #    buffer_format: BufferFormat,
    #    itemsize: int
    #) -> str:
    #    assert isinstance(buffer_format, StructuredBufferFormat)
    #    components: list[str] = []
    #    current_offset = 0
    #    for child, offset in zip(buffer_format._children_, buffer_format._offsets_, strict=True):
    #        assert isinstance(child, AtomicBufferFormat)
    #        if (padding := offset - current_offset):
    #            components.append(f"{padding}x")
    #        components.append(child._format_str_)
    #        current_offset = offset + child._nbytes_
    #    if (padding := itemsize - current_offset):
    #        components.append(f"{padding}x")
    #    components.append("/v")
    #    return " ".join(components)
