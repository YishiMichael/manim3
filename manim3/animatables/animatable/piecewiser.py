from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import Self

import attrs

from ...constants.custom_typing import (
    NP_xf8,
    NP_xi4
)


@attrs.frozen(kw_only=True)
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
