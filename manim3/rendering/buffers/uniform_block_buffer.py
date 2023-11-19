from __future__ import annotations


from typing import Self

import moderngl
import numpy as np

from ...constants.custom_typing import ShapeType
from ...lazy.lazy import Lazy
from ...toplevel.toplevel import Toplevel
#from ..buffer_format import StructuredBufferFormat
#from ..std140_layout import STD140Layout
from .buffer import Buffer
from ..field import (
    Field,
    StructuredField
)


class UniformBlockBuffer(Buffer):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        name: str,
        field_declarations: tuple[str, ...],
        structs: dict[str, tuple[str, ...]] | None = None,
        data_dict: dict[str, np.ndarray],
        array_lens: dict[str, int] | None = None
    ) -> None:
        super().__init__(
            array_lens=array_lens
        )
        self._name_ = name
        self._field_declarations_ = field_declarations
        if structs is not None:
            self._struct_items_ = tuple(structs.items())
        self._data_dict_ = data_dict

    @Lazy.variable()
    @staticmethod
    def _name_() -> str:
        return ""

    @Lazy.variable(plural=True)
    @staticmethod
    def _field_declarations_() -> tuple[str, ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _struct_items_() -> tuple[tuple[str, tuple[str, ...]], ...]:
        return ()

    @Lazy.variable()
    @staticmethod
    def _data_dict_() -> dict[str, np.ndarray]:
        return {}

    @Lazy.property()
    @staticmethod
    def _field_(
        name: str,
        field_declarations: tuple[str, ...],
        struct_items: tuple[tuple[str, tuple[str, ...]], ...],
        array_len_items: tuple[tuple[str, int], ...]
    ) -> StructuredField:
        return StructuredField(
            name=name,
            shape=(),
            fields=tuple(
                Field.parse_field(
                    field_declaration=field_declaration,
                    struct_dict=dict(struct_items),
                    array_lens_dict=dict(array_len_items)
                )
                for field_declaration in field_declarations
            )
        )

    @Lazy.property()
    @staticmethod
    def _buffer_(
        field: StructuredField,
        shape: ShapeType,
        data_dict: dict[str, np.ndarray]
    ) -> moderngl.Buffer:
        return Toplevel.context.buffer(field.write(shape, data_dict))

    #@Lazy.property()
    #@staticmethod
    #def _layout_() -> type[BufferLayout]:
    #    return Std140BufferLayout
