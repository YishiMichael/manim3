import itertools as it
from typing import Literal

import numpy as np

from ....constants.custom_typing import (
    NP_i4,
    NP_x2i4,
    NP_x3f8,
    NP_xi4,
    NP_xf8
)
from ....lazy.lazy import Lazy
from ....utils.space_utils import SpaceUtils
from ...mobject.mobject_attributes.mobject_attribute import (
    InterpolateHandler,
    MobjectAttribute
)


class Graph(MobjectAttribute):
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

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _edges_() -> NP_x2i4:
        return np.zeros((0, 2), dtype=np.int32)

    @classmethod
    def _interpolate(
        cls,
        graph_0: "Graph",
        graph_1: "Graph"
    ) -> "GraphInterpolateHandler":
        positions_0 = graph_0._positions_
        positions_1 = graph_1._positions_
        edges_0 = graph_0._edges_
        edges_1 = graph_1._edges_
        assert len(edges_0)
        assert len(edges_1)

        cumlengths_0 = Graph._get_cumlengths(positions_0, edges_0)
        cumlengths_1 = Graph._get_cumlengths(positions_1, edges_1)
        full_knots_0 = cumlengths_0 * cumlengths_1[-1]
        full_knots_1 = cumlengths_1 * cumlengths_0[-1]
        knots_0 = full_knots_0[1:-1]
        knots_1 = full_knots_1[1:-1]

        outline_edges_0, outline_positions_0, _ = Graph._get_decomposed_edges(
            positions=positions_0,
            edges=edges_0,
            insertions=np.arange(len(edges_1) - 1) + len(positions_0),
            full_knots=full_knots_0,
            values=knots_1,
            side="right"
        )
        outline_edges_1, outline_positions_1, _ = Graph._get_decomposed_edges(
            positions=positions_1,
            edges=edges_1,
            insertions=np.arange(len(edges_0) - 1) + len(positions_1),
            full_knots=full_knots_1,
            values=knots_0,
            side="left"
        )
        interpolated_positions_0, interpolated_positions_1, edges = Graph._get_unique_positions(
            positions_0=np.concatenate((
                positions_0,
                outline_positions_0
            )),
            positions_1=np.concatenate((
                positions_1,
                outline_positions_1
            )),
            edges_0=outline_edges_0,
            edges_1=outline_edges_1
        )
        return GraphInterpolateHandler(interpolated_positions_0, interpolated_positions_1, edges)

    @classmethod
    def _split(
        cls,
        graph: "Graph",
        alphas: NP_xf8
    ) -> "list[Graph]":
        #print(alphas)
        positions = graph._positions_
        edges = graph._edges_
        if not len(edges):
            return [Graph() for _ in range(len(alphas) + 1)]

        cumlengths = Graph._get_cumlengths(
            positions=positions,
            edges=edges
        )
        #n_positions = np.int32(len(positions))

        values = alphas * cumlengths[-1]
        interpolated_indices = np.searchsorted(cumlengths[1:-1], values)
        all_positions = np.concatenate((
            positions,
            Graph._interpolate_positions(
                positions=positions,
                edges=edges,
                full_knots=cumlengths,
                values=values,
                indices=interpolated_indices
            )
        ))
        #split_insertions = np.arange(len(interpolated_indices)) + len(positions)
        #prepends = np.insert(insertions, 0, edges[0, 0])
        #appends = np.append(insertions, edges[-1, 1])
        #print(edges)
        #print([Graph._reassemble_edges(
        #            edges=edges,
        #            transition_indices=np.arange(interpolated_index_0, interpolate_index_1),
        #            prepend=prepend,
        #            append=append,
        #            insertion_indices=np.zeros((0,), dtype=np.int32),
        #            insertions=np.zeros((0,), dtype=np.int32)
        #        )
        #for (prepend, append), (interpolated_index_0, interpolate_index_1) in zip(
        #        it.pairwise(np.array((edges[0, 0], *(np.arange(len(alphas)) + len(positions)), edges[-1, 1]))),
        #        it.pairwise(np.array((0, *interpolated_indices, len(edges) - 1))),
        #        strict=True
        #    )
        #])
        return [
            Graph(
                positions=all_positions,
                edges=Graph._reassemble_edges(
                    edges=edges,
                    transition_indices=np.arange(interpolated_index_0, interpolate_index_1),
                    prepend=prepend,
                    append=append,
                    insertion_indices=np.zeros((0,), dtype=np.int32),
                    insertions=np.zeros((0,), dtype=np.int32)
                )
            )
            for (prepend, append), (interpolated_index_0, interpolate_index_1) in zip(
                it.pairwise(np.array((edges[0, 0], *(np.arange(len(alphas)) + len(positions)), edges[-1, 1]))),
                it.pairwise(np.array((0, *interpolated_indices, len(edges) - 1))),
                strict=True
            )
        ]  # TODO: simplify: remove unused positions

    @classmethod
    def _concatenate(
        cls,
        graphs: "list[Graph]"
    ) -> "Graph":
        if not graphs:
            return Graph()

        positions = np.concatenate([
            graph._positions_
            for graph in graphs
        ])
        offsets = np.insert(np.cumsum([
            len(graph._positions_)
            for graph in graphs[:-1]
        ], dtype=np.int32), 0, 0)
        edges = np.concatenate([
            graph._edges_ + offset
            for graph, offset in zip(graphs, offsets, strict=True)
        ])
        return Graph(
            positions=positions,
            edges=edges
        )

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
        is_ring: bool  # TODO
    ) -> NP_x2i4:
        arange = np.arange(n)
        result = np.vstack((arange, np.roll(arange, -1))).T
        if not is_ring:
            result = result[:-1]
        return result


class GraphInterpolateHandler(InterpolateHandler[Graph]):
    __slots__ = (
        "_interpolated_positions_0",
        "_interpolated_positions_1",
        "_edges"
    )

    def __init__(
        self,
        interpolated_positions_0: NP_x3f8,
        interpolated_positions_1: NP_x3f8,
        edges: NP_x2i4
    ) -> None:
        super().__init__()
        self._interpolated_positions_0: NP_x3f8 = interpolated_positions_0
        self._interpolated_positions_1: NP_x3f8 = interpolated_positions_1
        self._edges: NP_x2i4 = edges

    def _interpolate(
        self,
        alpha: float
    ) -> Graph:
        return Graph(
            positions=SpaceUtils.lerp(self._interpolated_positions_0, self._interpolated_positions_1, alpha),
            edges=self._edges
        )
