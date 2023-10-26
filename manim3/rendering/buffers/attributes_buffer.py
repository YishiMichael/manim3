from __future__ import annotations


from typing import Self

import numpy as np

from ...lazy.lazy import Lazy
from ..buffer_layouts.buffer_layout import BufferLayout
from ..buffer_layouts.std140_buffer_layout import Std140BufferLayout
from .write_only_buffer import WriteOnlyBuffer


class AttributesBuffer(WriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        fields: list[str],
        num_vertex: int,
        array_lens: dict[str, int] | None = None,
        data: dict[str, np.ndarray]
    ) -> None:
        # Passing structs to an attribute is not allowed, so we eliminate the parameter `child_structs`.
        if array_lens is None:
            array_lens = {}
        super().__init__(
            field="__VertexStruct__ __vertex__[__NUM_VERTEX__]",
            child_structs={
                "__VertexStruct__": fields
            },
            array_lens={
                "__NUM_VERTEX__": num_vertex,
                **array_lens
            }
        )
        self.write(data)

    @Lazy.property()
    @staticmethod
    def _layout_() -> type[BufferLayout]:
        # Let's keep using std140 layout, hopefully giving a faster processing speed.
        return Std140BufferLayout
