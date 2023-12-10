from __future__ import annotations


from abc import (
    ABC,
    abstractmethod
)
from typing import Self

import attrs
import numpy as np

from ...constants.custom_typing import (
    NP_xf8,
    NP_xi4
)


@attrs.frozen(kw_only=True)
class PiecewiseInfo:
    split_alphas: NP_xf8
    concatenate_indices: NP_xi4


class Piecewiser(ABC):
    __slots__ = (
        "_backwards",
        "_n_segments"
    )

    def __init__(
        self: Self,
        n_segments: int,
        backwards: bool
    ) -> None:
        super().__init__()
        self._n_segments: int = n_segments
        self._backwards: bool = backwards

    def piecewise(
        self: Self,
        alpha: float
    ) -> PiecewiseInfo:
        n_segments = self._n_segments
        backwards = self._backwards
        start, stop = self.get_segment(alpha)
        if backwards:
            start, stop = -stop, -start
        start, stop = start % 1.0, start % 1.0 + float(np.clip(stop - start, 0.0, 1.0))

        if stop > 1.0:
            offset_0, offset_1 = stop - 1.0, start
            first_index = 0
        else:
            offset_0, offset_1 = start, stop
            first_index = 1

        split_alphas = np.column_stack(tuple(
            np.arange(n_segments, dtype=np.float64) + offset
            for offset in (offset_0, offset_1)
        )).flatten() / n_segments
        concatenate_indices = np.arange(first_index, 2 * n_segments + 1, 2)
        return PiecewiseInfo(
            split_alphas=split_alphas,
            concatenate_indices=concatenate_indices
        )

    @abstractmethod
    def get_segment(
        self: Self,
        alpha: float
    ) -> tuple[float, float]:
        pass
