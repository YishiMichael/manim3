from __future__ import annotations


from typing import Self

from ...constants.custom_typing import ShapeType
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject


class Buffer(LazyObject):
    __slots__ = ()

    def __init__(
        self: Self,
        shape: ShapeType | None = None,
        array_lens: dict[str, int] | None = None
    ) -> None:
        super().__init__()
        if shape is not None:
            self._shape_ = shape
        if array_lens is not None:
            self._array_len_items_ = tuple(array_lens.items())

    @Lazy.variable()
    @staticmethod
    def _shape_() -> ShapeType:
        return ()

    @Lazy.variable(plural=True)
    @staticmethod
    def _array_len_items_() -> tuple[tuple[str, int], ...]:
        return ()

    @Lazy.property(plural=True)
    @staticmethod
    def _macros_(
        array_len_items: tuple[tuple[str, int], ...]
    ) -> tuple[str, ...]:
        return tuple(
            f"#define {array_len_name} {array_len}"
            for array_len_name, array_len in array_len_items
        )
