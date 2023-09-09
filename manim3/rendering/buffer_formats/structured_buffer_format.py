import numpy as np

from ...lazy.lazy import Lazy
from .atomic_buffer_format import AtomicBufferFormat
from .buffer_format import BufferFormat
from .buffer_layout import BufferLayout


class StructuredBufferFormat(BufferFormat):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        children: list[BufferFormat],
        layout: BufferLayout
    ) -> None:
        structured_base_alignment = 16
        offsets: list[int] = []
        offset: int = 0
        for child in children:
            if layout == BufferLayout.STD140:
                if isinstance(child, AtomicBufferFormat):
                    base_alignment = child._base_alignment_
                elif isinstance(child, StructuredBufferFormat):
                    base_alignment = structured_base_alignment
                else:
                    raise TypeError
                offset += (-offset) % base_alignment
            offsets.append(offset)
            offset += child._nbytes_
        if layout == BufferLayout.STD140:
            offset += (-offset) % structured_base_alignment

        super().__init__(
            name=name,
            shape=shape
        )
        self._children_ = tuple(children)
        self._offsets_ = tuple(offsets)
        self._itemsize_ = offset

    @Lazy.variable_collection(hasher=Lazy.branch_hasher)
    @staticmethod
    def _children_() -> tuple[BufferFormat, ...]:
        return ()

    @Lazy.variable_collection(hasher=Lazy.naive_hasher)
    @staticmethod
    def _offsets_() -> tuple[int, ...]:
        return ()

    @Lazy.property(hasher=Lazy.naive_hasher)
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

    @Lazy.property_collection(hasher=Lazy.naive_hasher)
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
