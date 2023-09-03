from abc import (
    ABC,
    abstractmethod
)
from typing import (
    Generic,
    TypeVar
)

from ....constants.custom_typing import NP_xf8


_T = TypeVar("_T")


# TODO
class SplitHandler(Generic[_T], ABC):
    __slots__ = ()

    @abstractmethod
    def __init__(
        self,
        src: _T
    ) -> None:
        pass

    @abstractmethod
    def split(
        self,
        alpha_0: float,
        alpha_1: float
        #alphas: NP_xf8
    ) -> _T:
        pass
