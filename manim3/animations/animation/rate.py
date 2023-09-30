from abc import (
    ABC,
    abstractmethod
)

from ...constants.custom_typing import BoundaryT


class Rate(ABC):
    __slots__ = ()

    @abstractmethod
    def at(
        self,
        t: float
    ) -> float:
        pass

    @abstractmethod
    def at_boundary(
        self,
        boundary: BoundaryT
    ) -> BoundaryT:
        pass

    @abstractmethod
    def is_increasing(self) -> bool:
        pass
