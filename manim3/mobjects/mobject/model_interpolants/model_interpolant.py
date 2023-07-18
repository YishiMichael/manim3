from abc import ABC, abstractmethod

from ....constants.custom_typing import (
    NP_3f8,
    NP_44f8
)


class ModelInterpolant(ABC):
    __slots__ = ()

    @abstractmethod
    def __call__(
        self,
        alpha: float | NP_3f8 = 1.0
    ) -> NP_44f8:
        pass
