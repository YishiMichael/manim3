from __future__ import annotations


from typing import Self

import numpy as np

from ...lazy.lazy import Lazy
from ..buffer_layouts.buffer_layout import BufferLayout
from ..buffer_layouts.std140_buffer_layout import Std140BufferLayout
from .write_only_buffer import WriteOnlyBuffer


class UniformBlockBuffer(WriteOnlyBuffer):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        name: str,
        fields: list[str],
        child_structs: dict[str, list[str]] | None = None,
        array_lens: dict[str, int] | None = None,
        data: dict[str, np.ndarray]
    ) -> None:
        if child_structs is None:
            child_structs = {}
        super().__init__(
            field=f"__UniformBlockStruct__ {name}",
            child_structs={
                "__UniformBlockStruct__": fields,
                **child_structs
            },
            array_lens=array_lens
        )
        self.write(data)

    @Lazy.property()
    @staticmethod
    def _layout_() -> type[BufferLayout]:
        return Std140BufferLayout
