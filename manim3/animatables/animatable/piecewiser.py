from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from dataclasses import dataclass
from typing import Self

from ...constants.custom_typing import (
    NP_xf8,
    NP_xi4
)


@dataclass(
    frozen=True,
    kw_only=True,
    slots=True
)
class PiecewiseData:
    split_alphas: NP_xf8
    concatenate_indices: NP_xi4


class Piecewiser(ABC):
    __slots__ = ()

    @abstractmethod
    def piecewise(
        self: Self,
        alpha: float
    ) -> PiecewiseData:
        pass
