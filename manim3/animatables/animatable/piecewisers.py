from __future__ import annotations


from abc import abstractmethod
from typing import (
    Never,
    Self
)

import numpy as np

from .piecewiser import (
    PiecewiseData,
    Piecewiser
)


class EvenPiecewiser(Piecewiser):
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
    ) -> PiecewiseData:
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
        return PiecewiseData(
            split_alphas=split_alphas,
            concatenate_indices=concatenate_indices
        )

    @abstractmethod
    def get_segment(
        self: Self,
        alpha: float
    ) -> tuple[float, float]:
        pass


class PartialPiecewiser(EvenPiecewiser):
    __slots__ = ()

    def get_segment(
        self: Self,
        alpha: float
    ) -> tuple[float, float]:
        return (0.0, alpha)


class FlashPiecewiser(EvenPiecewiser):
    __slots__ = ("_proportion",)

    def __init__(
        self: Self,
        proportion: float,
        n_segments: int,
        backwards: bool
    ) -> None:
        super().__init__(
            n_segments=n_segments,
            backwards=backwards
        )
        self._proportion: float = proportion

    def get_segment(
        self: Self,
        alpha: float
    ) -> tuple[float, float]:
        proportion = self._proportion
        return (min(alpha * (1.0 + proportion), 1.0), max(alpha * (1.0 + proportion) - proportion, 0.0))


class DashedPiecewiser(EvenPiecewiser):
    __slots__ = ("_proportion",)

    def __init__(
        self: Self,
        proportion: float,
        n_segments: int,
        backwards: bool
    ) -> None:
        super().__init__(
            n_segments=n_segments,
            backwards=backwards
        )
        self._proportion: float = proportion

    def get_segment(
        self: Self,
        alpha: float
    ) -> tuple[float, float]:
        proportion = self._proportion
        return (alpha, alpha + proportion)


class Piecewisers:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def partial(
        cls: type[Self],
        n_segments: int = 1,
        backwards: bool = False
    ) -> PartialPiecewiser:
        return PartialPiecewiser(
            n_segments=n_segments,
            backwards=backwards
        )

    @classmethod
    def flash(
        cls: type[Self],
        proportion: float = 1.0 / 16,
        n_segments: int = 1,
        backwards: bool = False
    ) -> FlashPiecewiser:
        return FlashPiecewiser(
            proportion=proportion,
            n_segments=n_segments,
            backwards=backwards
        )

    @classmethod
    def dashed(
        cls: type[Self],
        proportion: float = 1.0 / 2,
        n_segments: int = 16,
        backwards: bool = False
    ) -> DashedPiecewiser:
        return DashedPiecewiser(
            proportion=proportion,
            n_segments=n_segments,
            backwards=backwards
        )
