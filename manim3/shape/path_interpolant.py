import itertools as it
from typing import Literal

import numpy as np

from ..constants.custom_typing import (
    NP_xf8,
    NP_xu4
)
from ..lazy.lazy import LazyObject


class PathInterpolant(LazyObject):
    __slots__ = ()

    @classmethod
    def _lengths_to_knots(
        cls,
        lengths: NP_xf8
    ) -> NP_xf8:
        assert len(lengths)
        lengths_cumsum = np.maximum(lengths, 1e-6).cumsum()
        return np.insert(lengths_cumsum / lengths_cumsum[-1], 0, 0.0)

    @classmethod
    def _interpolate_knots(
        cls,
        knots: NP_xf8,
        values: NP_xf8,
        *,
        side: Literal["left", "right"]
    ) -> tuple[NP_xu4, NP_xf8]:
        index = (np.searchsorted(knots, values, side=side) - 1).astype(np.uint32)
        residue = (values - knots[index]) / (knots[index + 1] - knots[index])
        return index, residue

    @classmethod
    def _partial_residues(
        cls,
        knots: NP_xf8,
        start: float,
        stop: float
    ) -> tuple[int, float, int, float]:
        start_index, start_residue = cls._interpolate_knots(knots, start * np.ones((1,)), side="right")
        stop_index, stop_residue = cls._interpolate_knots(knots, stop * np.ones((1,)), side="left")
        return int(start_index), float(start_residue), int(stop_index), float(stop_residue)

    @classmethod
    def _zip_residues_list(
        cls,
        *knots_tuple: NP_xf8
    ) -> tuple[list[NP_xf8], ...]:
        all_knots, unique_inverse = np.unique(np.concatenate(knots_tuple), return_inverse=True)
        offsets = np.insert(np.cumsum([
            len(knots) for knots in knots_tuple
        ]), 0, 0)
        return tuple(
            [
                (all_knots[start_index:stop_index + 1] - all_knots[start_index]) / (all_knots[stop_index] - all_knots[start_index])
                for start_index, stop_index in it.pairwise(unique_inverse[start_offset:stop_offset])
            ]
            for start_offset, stop_offset in it.pairwise(offsets)
        )
