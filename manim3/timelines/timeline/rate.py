from __future__ import annotations


from abc import abstractmethod
from typing import Self

from ...constants.custom_typing import BoundaryType
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject


class Rate(LazyObject):
    __slots__ = ()

    @Lazy.property()
    @staticmethod
    def _is_increasing_() -> bool:
        return True

    @Lazy.property()
    @staticmethod
    def _boundaries_() -> tuple[BoundaryType, BoundaryType]:
        return (0, 1)

    @abstractmethod
    def at(
        self: Self,
        t: float
    ) -> float:
        pass
