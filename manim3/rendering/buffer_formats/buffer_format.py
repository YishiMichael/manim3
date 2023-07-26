import operator as op
from functools import reduce

import numpy as np

from ...lazy.lazy import (
    Lazy,
    LazyObject
)


class BufferFormat(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...]
    ) -> None:
        super().__init__()
        self._name_ = name
        self._shape_ = shape

    @Lazy.variable_hashable
    @classmethod
    def _name_(cls) -> str:
        return ""

    @Lazy.variable_hashable
    @classmethod
    def _shape_(cls) -> tuple[int, ...]:
        return ()

    @Lazy.variable_hashable
    @classmethod
    def _itemsize_(cls) -> int:
        # Implemented in subclasses.
        return 0

    @Lazy.property_hashable
    @classmethod
    def _size_(
        cls,
        shape: tuple[int, ...]
    ) -> int:
        return reduce(op.mul, shape, 1)

    @Lazy.property_hashable
    @classmethod
    def _nbytes_(
        cls,
        itemsize: int,
        size: int
    ) -> int:
        return itemsize * size

    @Lazy.property_hashable
    @classmethod
    def _is_empty_(
        cls,
        size: int
    ) -> bool:
        return not size

    @Lazy.property_hashable
    @classmethod
    def _dtype_(cls) -> np.dtype:
        # Implemented in subclasses.
        return np.dtype("f4")
