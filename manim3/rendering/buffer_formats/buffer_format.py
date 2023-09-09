import operator as op
from functools import reduce

import numpy as np

from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject


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

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _name_() -> str:
        return ""

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _shape_() -> tuple[int, ...]:
        return ()

    @Lazy.variable(hasher=Lazy.naive_hasher)
    @staticmethod
    def _itemsize_() -> int:
        # Implemented in subclasses.
        return 0

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _size_(
        shape: tuple[int, ...]
    ) -> int:
        return reduce(op.mul, shape, 1)

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _nbytes_(
        itemsize: int,
        size: int
    ) -> int:
        return itemsize * size

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_empty_(
        size: int
    ) -> bool:
        return not size

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _dtype_() -> np.dtype:
        # Implemented in subclasses.
        return np.dtype("f4")
