#from __future__ import annotations


#from typing import Self

#from .buffer_layout import BufferLayout


#class Std140BufferLayout(BufferLayout):
#    __slots__ = ()

#    @classmethod
#    def _get_atomic_col_padding(
#        cls: type[Self],
#        col_len: int,
#        row_len: int,
#        is_array: bool
#    ) -> int:
#        return 0 if not is_array and row_len == 1 else 4 - col_len

#    @classmethod
#    def _get_atomic_base_alignment(
#        cls: type[Self],
#        col_len: int,
#        row_len: int,
#        base_itemsize: int,
#        is_array: bool
#    ) -> int:
#        return (col_len if not is_array and col_len <= 2 and row_len == 1 else 4) * base_itemsize

#    @classmethod
#    def _get_structured_base_alignment(
#        cls: type[Self]
#    ) -> int:
#        return 16
