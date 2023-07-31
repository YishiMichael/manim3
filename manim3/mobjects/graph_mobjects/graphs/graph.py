from typing import Literal

import numpy as np

from ....constants.custom_typing import (
    NP_i4,
    NP_x2i4,
    NP_x3f8,
    NP_xi4,
    NP_xf8
)
from ....lazy.lazy import (
    Lazy,
    LazyObject
)
from ....utils.space_utils import SpaceUtils


class Graph(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x3f8 | None = None,
        edges: NP_x2i4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if edges is not None:
            self._edges_ = edges

    @Lazy.variable_array
    @classmethod
    def _positions_(cls) -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable_array
    @classmethod
    def _edges_(cls) -> NP_x2i4:
        return np.zeros((0, 2), dtype=np.int32)

    @classmethod
    def _get_cumlengths(
        cls,
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> NP_xf8:
        lengths = SpaceUtils.norm(positions[edges[:, 1]] - positions[edges[:, 0]])
        return np.insert(np.cumsum(lengths), 0, 0.0)

    @classmethod
    def _get_disjoints(
        cls,
        edges: NP_x2i4
    ) -> NP_xi4:
        return np.flatnonzero(edges[:-1, 1] - edges[1:, 0])

    @classmethod
    def _interpolate_positions(
        cls,
        positions: NP_x3f8,
        edges: NP_x2i4,
        full_knots: NP_xf8,
        values: NP_xf8,
        indices: NP_xi4
    ) -> NP_x3f8:
        residues = (values - full_knots[indices]) / np.maximum(full_knots[indices + 1] - full_knots[indices], 1e-6)
        return SpaceUtils.lerp(
            positions[edges[indices, 0]],
            positions[edges[indices, 1]],
            residues[:, None]
        )

    @classmethod
    def _reassemble_edges(
        cls,
        edges: NP_x2i4,
        transition_indices: NP_xi4,
        prepend: NP_i4,
        append: NP_i4,
        insertion_indices: NP_xi4,
        insertions: NP_xi4
    ) -> NP_x2i4:
        return np.column_stack((
            np.insert(np.insert(
                edges[transition_indices + 1, 0],
                insertion_indices,
                insertions
            ), 0, prepend),
            np.append(np.insert(
                edges[transition_indices, 1],
                insertion_indices,
                insertions
            ), append)
        ))

    @classmethod
    def _get_decomposed_edges(
        cls,
        positions: NP_x3f8,
        edges: NP_x2i4,
        insertions: NP_xi4,
        full_knots: NP_xf8,
        values: NP_xf8,
        side: Literal["left", "right"]
    ) -> tuple[NP_x2i4, NP_x3f8, NP_xi4]:
        interpolated_indices = np.searchsorted(full_knots[1:-1], values, side=side).astype(np.int32)
        decomposed_edges = cls._reassemble_edges(
            edges=edges,
            transition_indices=np.arange(len(edges) - 1),
            prepend=edges[0, 0],
            append=edges[-1, 1],
            insertion_indices=interpolated_indices,
            insertions=insertions
        )
        interpolated_positions = cls._interpolate_positions(
            positions=positions,
            edges=edges,
            full_knots=full_knots,
            values=values,
            indices=interpolated_indices
        )
        return decomposed_edges, interpolated_positions, interpolated_indices

    @classmethod
    def _get_unique_positions(
        cls,
        positions_0: NP_x3f8,
        positions_1: NP_x3f8,
        edges_0: NP_x2i4,
        edges_1: NP_x2i4
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        unique_edges, edges_inverse = np.unique(
            np.array((edges_0.flatten(), edges_1.flatten())),
            axis=1,
            return_inverse=True
        )
        return (
            positions_0[unique_edges[0]],
            positions_1[unique_edges[1]],
            edges_inverse.reshape((-1, 2))
        )

    @classmethod
    def _get_consecutive_edges(
        cls,
        n: int,
        *,
        is_ring: bool
    ) -> NP_x2i4:
        arange = np.arange(n)
        result = np.vstack((arange, np.roll(arange, -1))).T
        if not is_ring:
            result = result[:-1]
        return result
