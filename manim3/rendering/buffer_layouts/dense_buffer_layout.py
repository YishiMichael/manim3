from __future__ import annotations


from typing import Self

from .buffer_layout import BufferLayout


class DenseBufferLayout(BufferLayout):
    __slots__ = ()

    @classmethod
    def _get_atomic_col_padding(
        cls: type[Self],
        shape: tuple[int, ...],
        col_len: int,
        row_len: int
    ) -> int:
        return 0

    @classmethod
    def _get_atomic_base_alignment(
        cls: type[Self],
        shape: tuple[int, ...],
        col_len: int,
        row_len: int,
        base_itemsize: int
    ) -> int:
        return 1

    @classmethod
    def _get_structured_base_alignment(
        cls: type[Self]
    ) -> int:
        return 1
