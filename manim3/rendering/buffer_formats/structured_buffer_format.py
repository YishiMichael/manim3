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
        self._children_.reset(children)
        self._offsets_ = tuple(offsets)
        self._itemsize_ = offset

    @Lazy.variable_collection
    @classmethod
    def _children_(cls) -> list[BufferFormat]:
        return []

    @Lazy.variable_hashable
    @classmethod
    def _offsets_(cls) -> tuple[int, ...]:
        return ()

    @Lazy.property_hashable
    @classmethod
    def _dtype_(
        cls,
        children__name: list[str],
        children__dtype: list[np.dtype],
        children__shape: list[tuple[int, ...]],
        offsets: tuple[int, ...],
        itemsize: int
    ) -> np.dtype:
        return np.dtype({
            "names": children__name,
            "formats": list(zip(children__dtype, children__shape, strict=True)),
            "offsets": list(offsets),
            "itemsize": itemsize
        })
