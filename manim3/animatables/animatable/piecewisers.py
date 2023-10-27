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
        #center = self.get_center(alpha) * (-1.0 if backwards else 1.0)
        #length = float(np.clip(self.get_length(alpha), 0.0, 1.0))
        #start = center - length / 2.0
        #stop = center + length / 2.0
        start, stop = self.get_segment(alpha)
        assert 0.0 <= stop - start <= 1.0
        if backwards:
            start, stop = -stop, -start

        start_offset = start % 1.0
        stop_offset = start_offset + (stop - start)
        if stop_offset > 1.0:
            start_offset, stop_offset = stop_offset - 1.0, start_offset
            first_index = 0
        else:
            first_index = 1

        split_alphas = np.column_stack(tuple(
            np.arange(n_segments, dtype=np.float64) + offset
            for offset in (start_offset, stop_offset)
        )).flatten() / n_segments
        concatenate_indices = np.arange(first_index, 2 * n_segments + 1, 2)
        #start_floor = np.floor(start)
        #stop_floor = np.floor(stop)
        #split_starts = np.linspace(0.0, 1.0, n_segments, endpoint=False) + start_offset
        #split_stops = np.linspace(0.0, 1.0, n_segments, endpoint=False) + stop_offset
        #split_start_cutoff = int(start % n_segments)
        #split_stops = np.linspace(stop, stop + 1.0, n_segments, endpoint=False)
        #boundaries = np.array((start, stop))
        #wrap_flag = int(stop) - int(start)
        ##if backwards:
        ##    wrap_flag = 1 - wrap_flag
        #wrap_shift = 2 * int(start % n_segments) + wrap_flag
        #split_alphas = np.linspace(boundaries, boundaries + 1.0, n_segments, endpoint=False).flatten()
        #split_alphas = np.roll(split_alphas, wrap_shift)
        #split_alphas[:wrap_shift] -= np.floor(stop)
        #split_alphas[wrap_shift:] -= np.floor(start)
        #concatenate_indices = np.arange(1 - wrap_flag, 2 * n_segments + 1, 2)
        #if backwards:
        #    split_alphas = 1.0 - split_alphas[::-1]
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
        assert proportion >= 0.0
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
        assert proportion >= 0.0
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
