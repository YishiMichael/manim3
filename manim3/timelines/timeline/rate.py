from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import Self

from ...constants.custom_typing import BoundaryT


class Rate(ABC):
    __slots__ = ()

    @abstractmethod
    def at(
        self: Self,
        t: float
    ) -> float:
        pass

    @abstractmethod
    def at_boundary(
        self: Self,
        boundary: BoundaryT
    ) -> BoundaryT:
        pass

    @abstractmethod
    def is_increasing(
        self: Self
    ) -> bool:
        pass
