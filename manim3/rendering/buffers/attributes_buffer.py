from __future__ import annotations


from typing import Self

import moderngl
import numpy as np

from ...constants.custom_typing import (
    NP_xi4,
    ShapeType
)
from ...lazy.lazy import Lazy
from ...toplevel.toplevel import Toplevel
from ..mgl_enums import PrimitiveMode
from ..field import (
    AtomicField,
    Field,
    StructuredField
)
from .buffer import Buffer


class AttributesBuffer(Buffer):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        field_declarations: tuple[str, ...],
        data_dict: dict[str, np.ndarray],
        index: NP_xi4 | None = None,
        primitive_mode: PrimitiveMode,
        vertices_count: int,
        array_lens: dict[str, int] | None = None
    ) -> None:
        super().__init__(
            shape=(vertices_count,),
            array_lens=array_lens
        )
        self._field_declarations_ = field_declarations
        self._data_dict_ = data_dict
        self._vertices_count_ = vertices_count
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
    def _vertices_count_() -> int:
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
    def _buffer_(
        merged_field: StructuredField,
        shape: ShapeType,
        data_dict: dict[str, np.ndarray]
    ) -> moderngl.Buffer:
        return Toplevel._get_context().buffer(merged_field.write(shape, data_dict))

    @Lazy.property()
    @staticmethod
    def _index_buffer_(
        index_bytes: bytes,
        use_index_buffer: bool
    ) -> moderngl.Buffer | None:
        if not use_index_buffer:
            return None
        return Toplevel._get_context().buffer(index_bytes)
