from __future__ import annotations


from typing import Self

import numpy as np

from ...lazy.lazy import Lazy
from .buffer_format import BufferFormat


class StructuredBufferFormat(BufferFormat):
    __slots__ = ()

    def __init__(
        self: Self,
        *,
        name: str,
        shape: tuple[int, ...],
        children: tuple[BufferFormat, ...],
        offsets: tuple[int, ...],
        itemsize: int,
        base_alignment: int
        #children: list[BufferFormat],
        #layout: BufferLayout
    ) -> None:
        #if layout == BufferLayout.STD140:
        #    base_alignment = 16
        #else:
        #    base_alignment = 1

        #offsets: list[int] = []
        #offset: int = 0
        #for child in children:
        #    offset += (-offset) % child._base_alignment_
        #    offsets.append(offset)
        #    offset += child._nbytes_
        #offset += (-offset) % base_alignment

        super().__init__(
            name=name,
            shape=shape
        )
        self._children_ = children
        self._offsets_ = offsets
        self._itemsize_ = itemsize
        self._base_alignment_ = base_alignment

    @Lazy.variable(plural=True)
    @staticmethod
    def _children_() -> tuple[BufferFormat, ...]:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _offsets_() -> tuple[int, ...]:
        return ()

    @Lazy.property()
    @staticmethod
    def _dtype_(
        children__name: tuple[str, ...],
        children__dtype: tuple[np.dtype, ...],
        children__shape: tuple[tuple[int, ...], ...],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> np.dtype:
        return np.dtype({
            "names": children__name,
            "formats": list(zip(children__dtype, children__shape, strict=True)),
            "offsets": list(offsets),
            "itemsize": itemsize
        })

    @Lazy.property(plural=True)
    @staticmethod
    def _pointers_(
        children__name: tuple[str, ...],
        children__pointers: tuple[tuple[tuple[tuple[str, ...], int], ...], ...]
    ) -> tuple[tuple[tuple[str, ...], int], ...]:
        return tuple(
            ((child_name,) + name_chain, base_ndim)
            for child_name, child_pointers in zip(children__name, children__pointers, strict=True)
            for name_chain, base_ndim in child_pointers
        )
