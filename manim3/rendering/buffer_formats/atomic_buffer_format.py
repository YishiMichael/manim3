from __future__ import annotations


from typing import Self

import numpy as np

from ...lazy.lazy import Lazy
from .buffer_format import BufferFormat


class AtomicBufferFormat(BufferFormat):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        name: str,
        shape: tuple[int, ...],
        base_char: str,
        base_itemsize: int,
        base_ndim: int,
        row_len: int,
        col_len: int,
        col_padding: int,
        itemsize: int,
        base_alignment: int
        #gl_dtype_str: str,
        #layout: BufferLayout
    ) -> None:
        #base_char, base_itemsize, base_shape = type(self)._GL_DTYPES[gl_dtype_str]
        #assert len(base_shape) <= 2 and all(2 <= l <= 4 for l in base_shape)
        #shape_dict = dict(enumerate(base_shape))
        #col_len = shape_dict.get(0, 1)
        #row_len = shape_dict.get(1, 1)
        #if layout == BufferLayout.STD140:
        #    col_padding = 0 if not shape and row_len == 1 else 4 - col_len
        #    base_alignment = (col_len if not shape and col_len <= 2 and row_len == 1 else 4) * base_itemsize
        #else:
        #    col_padding = 0
        #    base_alignment = 1

        super().__init__(
            name=name,
            shape=shape
        )
        #self._base_char_ = base_char
        #self._base_itemsize_ = base_itemsize
        #self._base_ndim_ = len(base_shape)
        #self._row_len_ = row_len
        #self._col_len_ = col_len
        #self._col_padding_ = col_padding
        #self._itemsize_ = row_len * (col_len + col_padding) * base_itemsize
        #self._base_alignment_ = base_alignment
        self._base_char_ = base_char
        self._base_itemsize_ = base_itemsize
        self._base_ndim_ = base_ndim
        self._row_len_ = row_len
        self._col_len_ = col_len
        self._col_padding_ = col_padding
        self._itemsize_ = itemsize
        self._base_alignment_ = base_alignment

    @Lazy.variable()
    @staticmethod
    def _base_char_() -> str:
        return ""

    @Lazy.variable()
    @staticmethod
    def _base_itemsize_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _base_ndim_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _row_len_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _col_len_() -> int:
        return 0

    @Lazy.variable()
    @staticmethod
    def _col_padding_() -> int:
        return 0

    @Lazy.property()
    @staticmethod
    def _dtype_(
        base_char: str,
        base_itemsize: int,
        col_len: int,
        col_padding: int,
        row_len: int
    ) -> np.dtype:
        return np.dtype((np.dtype({
            "names": ["_"],
            "formats": [(np.dtype(f"{base_char}{base_itemsize}"), (col_len,))],
            "itemsize": (col_len + col_padding) * base_itemsize
        }), (row_len,)))

    @Lazy.property(plural=True)
    @staticmethod
    def _pointers_(
        base_ndim: int
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        return (((), base_ndim),)

    @Lazy.property()
    @staticmethod
    def _format_str_(
        col_len: int,
        base_char: str,
        base_itemsize: int,
        col_padding: int,
        row_len: int,
        size: int
    ) -> str:
        row_components = [f"{col_len}{base_char}{base_itemsize}"]
        if col_padding:
            row_components.append(f"{col_padding}x{base_itemsize}")
        return " ".join(row_components * (row_len * size))
