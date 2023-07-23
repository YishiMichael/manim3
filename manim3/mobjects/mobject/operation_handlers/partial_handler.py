from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Generic,
    TypeVar
)


_T = TypeVar("_T")


class PartialHandler(Generic[_T], ABC):
    __slots__ = ()

    @abstractmethod
    def __init__(
        self,
        src: _T
    ) -> None:
        pass

    @abstractmethod
    def partial(
        self,
        alpha_0: float,
        alpha_1: float
    ) -> _T:
        pass
