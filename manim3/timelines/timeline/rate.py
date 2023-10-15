from __future__ import annotations


from abc import abstractmethod
from typing import Self

from ...constants.custom_typing import BoundaryT
from ...lazy.lazy import Lazy
from ...lazy.lazy_object import LazyObject


class Rate(LazyObject):
    __slots__ = ()

    #def __init__(
    #    self: Self,
    #    *,
    #    boundary_0: BoundaryT,
    #    boundary_1: BoundaryT,
    #    is_increasing: bool
    #) -> None:
    #    super().__init__()
    #    self._boundary_0: BoundaryT = boundary_0
    #    self._boundary_1: BoundaryT = boundary_1
    #    self._is_increasing: bool = is_increasing

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _is_increasing_() -> bool:
        return True

    @Lazy.property(hasher=Lazy.naive_hasher)
    @staticmethod
    def _boundaries_() -> tuple[BoundaryT, BoundaryT]:
        return (0, 1)

    @abstractmethod
    def at(
        self: Self,
        t: float
    ) -> float:
        pass

    #@abstractmethod
    #def at_boundary(
    #    self: Self,
    #    boundary: BoundaryT
    #) -> BoundaryT:
    #    pass

    #@abstractmethod
    #def is_increasing(
    #    self: Self
    #) -> bool:
    #    pass
