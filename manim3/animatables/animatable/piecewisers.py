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
        segment_center = self.get_segment_center(alpha) * (-1.0 if backwards else 1.0)
        segment_length = float(np.clip(self.get_segment_length(alpha), 0.0, 1.0))
        segment_start = segment_center - segment_length / 2.0
        segment_stop = segment_center + segment_length / 2.0

        boundaries = np.array((segment_start, segment_stop))
        wrap_flag = int(segment_stop) - int(segment_start)
        #if backwards:
        #    wrap_flag = 1 - wrap_flag
        wrap_shift = 2 * int(segment_start % n_segments) + wrap_flag
        split_alphas = np.linspace(boundaries, boundaries + 1.0, n_segments, endpoint=False).flatten()
        split_alphas = np.roll(split_alphas, wrap_shift)
        split_alphas[:wrap_shift] -= np.floor(segment_stop)
        split_alphas[wrap_shift:] -= np.floor(segment_start)
        concatenate_indices = np.arange(1 - wrap_flag, 2 * n_segments + 1, 2)
        #if backwards:
        #    split_alphas = 1.0 - split_alphas[::-1]
        print(segment_center, segment_length, split_alphas, concatenate_indices)
        return PiecewiseData(
            split_alphas=split_alphas,
            concatenate_indices=concatenate_indices
        )

    @abstractmethod
    def get_segment_center(
        self: Self,
        alpha: float
    ) -> float:
        pass

    @abstractmethod
    def get_segment_length(
        self: Self,
        alpha: float
    ) -> float:
        pass


class PartialPiecewiser(EvenPiecewiser):
    __slots__ = ()

    def get_segment_center(
        self: Self,
        alpha: float
    ) -> float:
        return alpha / 2.0

    def get_segment_length(
        self: Self,
        alpha: float
    ) -> float:
        return alpha


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

    def get_segment_center(
        self: Self,
        alpha: float
    ) -> float:
        proportion = self._proportion
        return (min(alpha * (1.0 + proportion), 1.0) + max(alpha * (1.0 + proportion) - proportion, 0.0)) / 2.0

    def get_segment_length(
        self: Self,
        alpha: float
    ) -> float:
        proportion = self._proportion
        return min(alpha * (1.0 + proportion), 1.0) - max(alpha * (1.0 + proportion) - proportion, 0.0)


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

    def get_segment_center(
        self: Self,
        alpha: float
    ) -> float:
        return alpha

    def get_segment_length(
        self: Self,
        alpha: float
    ) -> float:
        return self._proportion


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
