from typing import Literal

import numpy as np

from ....constants.custom_typing import (
    NP_x2i4,
    NP_x3f8,
    NP_xi4,
    NP_xf8
)
from ....lazy.lazy import (
    Lazy,
    LazyObject
)
from ....utils.space import SpaceUtils


class Graph(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x3f8 | None = None,
        indices: NP_x2i4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if indices is not None:
            self._indices_ = indices

    @Lazy.variable_array
    @classmethod
    def _positions_(cls) -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable_array
    @classmethod
    def _indices_(cls) -> NP_x2i4:
        return np.zeros((0, 2), dtype=np.int32)

    @Lazy.property_array
    @classmethod
    def _knots_(
        cls,
        positions: NP_x3f8,
        indices: NP_x2i4
    ) -> NP_xf8:
        lengths = SpaceUtils.norm(positions[indices[:, 1]] - positions[indices[:, 0]])
        return np.insert(np.cumsum(lengths), 0, 0.0)

    @classmethod
    def _interpolate_knots(
        cls,
        knots: NP_xf8,
        values: NP_xf8,
        *,
        side: Literal["left", "right"] = "left"
    ) -> tuple[NP_xi4, NP_x3f8]:
        indices = np.clip(np.searchsorted(knots, values, side=side), 1, len(knots) - 1, dtype=np.int32) - 1
        residues = (values - knots[indices]) / np.maximum(knots[indices + 1] - knots[indices], 1e-6)
        return indices, residues

    @classmethod
    def _get_consecutive_indices(
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

    #@classmethod
    #def partial(
    #    cls,
    #    graph: "Graph"
    #) -> "Callable[[float, float], Graph]":
    #    positions = graph._positions_
    #    indices = graph._indices_
    #    knots = graph._knots_
    #    n_positions = len(positions)
    #    length = knots[-1]

    #    def callback(
    #        alpha_0: float,
    #        alpha_1: float
    #    ) -> Graph:
    #        if alpha_0 > alpha_1:
    #            return Graph()
    #        interpolated_indices, residues = cls._interpolate_knots(knots, np.array((alpha_0, alpha_1)) * length)
    #        extended_positions = np.concatenate((
    #            positions,
    #            SpaceUtils.lerp(
    #                positions[indices[interpolated_indices - 1]],
    #                positions[indices[interpolated_indices]],
    #                residues[:, None]
    #            )
    #            #np.array((
    #            #    SpaceUtils.lerp(
    #            #        positions[indices[2 * interpolate_index_0]], positions[indices[2 * interpolate_index_0 + 1]]
    #            #    )(residue_0),
    #            #    SpaceUtils.lerp(
    #            #        positions[indices[2 * interpolate_index_1]], positions[indices[2 * interpolate_index_1 + 1]]
    #            #    )(residue_1)
    #            #))
    #        ))
    #        #interpolated_index_0, residue_0 = cls._interpolate_knots(knots, alpha_0 * length, side="right")
    #        #interpolated_index_1, residue_1 = cls._interpolate_knots(knots, alpha_1 * length, side="left")
    #        return Graph(
    #            positions=extended_positions,
    #            indices=np.insert(
    #                np.array((n_positions, n_positions + 1)),
    #                1,
    #                indices[slice(*interpolated_indices)]
    #            )
    #        )

    #    return callback

    #@classmethod
    #def interpolate(
    #    cls,
    #    graph_0: "Graph",
    #    graph_1: "Graph"
    #) -> "Callable[[float], Graph]":



    #    #interpolated_positions_0 = SpaceUtils.lerp(positions_0[indices_0 - 1], positions_0[indices_0])(residues_0[:, None])
    #    #interpolated_positions_1 = SpaceUtils.lerp(positions_1[indices_1 - 1], positions_1[indices_1])(residues_1[:, None])

    #    #edge_indices_0 = np.array((edges_0[:-1, 1], edges_0[1:, 0]))
    #    #joint_booleans_0 = edge_indices_0[0] == edge_indices_0[1]
    #    #preserved_vertex_indices_0 = np.delete(edge_indices_0.flatten(), 2 * joint_booleans_0.nonzero()[0])
    #    #repeats_0 = np.where(joint_booleans_0, 1, 2)

    #    #edge_indices_1 = np.array((edges_1[:-1, 1], edges_1[1:, 0]))
    #    #joint_booleans_1 = edge_indices_1[0] == edge_indices_1[1]
    #    #preserved_vertex_indices_1 = np.delete(edge_indices_1.flatten(), 2 * joint_booleans_1.nonzero()[0])
    #    #repeats_1 = np.where(joint_booleans_1, 1, 2)

    #    ## Merge together boundaries if they match.
    #    #boundary_vertex_indices = np.array(((edges_0[0, 0], edges_1[0, 0]), (edges_0[-1, 1], edges_1[-1, 1])))
    #    #ring_boolean = np.all(boundary_vertex_indices[0] == boundary_vertex_indices[1])
    #    ## `preserved_boundary_indices.shape` is either `(2, 2)` or `(1, 2)`.
    #    #preserved_boundary_indices = np.delete(boundary_vertex_indices, np.atleast_1d(ring_boolean).nonzero()[0], axis=0)

    #    #extended_positions_0 = np.concatenate((
    #    #    np.repeat(interpolated_positions_0, repeats_1, axis=0),
    #    #    positions_0[preserved_vertex_indices_0],
    #    #    positions_0[preserved_boundary_indices[:, 0]]
    #    #))
    #    #extended_positions_1 = np.concatenate((
    #    #    positions_1[preserved_vertex_indices_1],
    #    #    np.repeat(interpolated_positions_1, repeats_0, axis=0),
    #    #    positions_1[preserved_boundary_indices[:, 1]]
    #    #))
    #    #concatenated_indices = np.concatenate((
    #    #    np.repeat(indices_0 + np.arange(len(indices_0)), repeats_1),
    #    #    np.repeat(indices_1 + np.arange(len(indices_1)), repeats_0)
    #    #))
    #    #edges = np.array((
    #    #    len(concatenated_indices),
    #    #    *np.repeat(
    #    #        np.argsort(concatenated_indices),
    #    #        np.where(np.concatenate((
    #    #            np.repeat(joint_booleans_1, repeats_1),
    #    #            np.repeat(joint_booleans_0, repeats_0)
    #    #        ))[concatenated_indices], 2, 1)
    #    #    ),
    #    #    len(concatenated_indices) + len(preserved_boundary_indices) - 1
    #    #)).reshape((-1, 2))

    #    def callback(
    #        alpha: float
    #    ) -> Graph:
    #        return Graph(
    #            positions=SpaceUtils.lerp(aligned_positions_0, aligned_positions_1)(alpha),
    #            indices=indices
    #        )

    #    return callback

    #@classmethod
    #def _interpolate(
    #    cls,
    #    stroke_0: "Stroke",
    #    stroke_1: "Stroke",
    #    *,
    #    has_inlay: bool
    #) -> "Callable[[float], Stroke]":
    #    assert len(stroke_0._points_)
    #    assert len(stroke_1._points_)
    #    points_0 = stroke_0._points_
    #    points_1 = stroke_1._points_
    #    disjoints_0 = stroke_0._disjoints_
    #    disjoints_1 = stroke_1._disjoints_
    #    knots_0 = stroke_0._cumlengths_ * stroke_1._length_
    #    knots_1 = stroke_1._cumlengths_ * stroke_0._length_

    #    #indices_0 = np.minimum(np.searchsorted(knots_0, knots_1[:-1], side="right"), len(knots_0) - 1)
    #    #indices_1 = np.maximum(np.searchsorted(knots_1, knots_0[1:], side="left"), 1)
    #    #residues_0 = (knots_1[:-1] - knots_0[indices_0 - 1]) / np.maximum(knots_0[indices_0] - knots_0[indices_0 - 1], 1e-6)
    #    #residues_1 = (knots_0[1:] - knots_1[indices_1 - 1]) / np.maximum(knots_1[indices_1] - knots_1[indices_1 - 1], 1e-6)
    #    indices_0, residues_0 = cls._interpolate_knots(knots_0, knots_1[1:-1], side="right")
    #    indices_1, residues_1 = cls._interpolate_knots(knots_1, knots_0[1:-1], side="left")

    #    total_indices_0 = indices_0 + np.arange(len(indices_0)) - 1
    #    total_indices_1 = indices_1 + np.arange(len(indices_1))
    #    points_order = np.argsort(np.concatenate((total_indices_0, total_indices_1)))
    #    total_points_0 = np.concatenate((
    #        SpaceUtils.lerp(points_0[indices_0 - 1], points_0[indices_0])(residues_0[:, None]),
    #        points_0[1:]
    #    ))[points_order]
    #    total_points_1 = np.concatenate((
    #        points_1[:-1],
    #        SpaceUtils.lerp(points_1[indices_1 - 1], points_1[indices_1])(residues_1[:, None])
    #    ))[points_order]

    #    total_disjoints_0 = total_indices_1[disjoints_0 - 1]
    #    total_disjoints_1 = total_indices_0[disjoints_1]
    #    total_disjoints = np.sort(np.concatenate((total_disjoints_0, total_disjoints_1)))

    #    if has_inlay:
    #        n_points = len(indices_0) + len(indices_1)
    #        disjoint_indices_0 = np.searchsorted(total_disjoints_0, total_disjoints_1, side="right")
    #        disjoint_indices_1 = np.searchsorted(total_disjoints_1, total_disjoints_0, side="left")
    #        inlay_points_list_0 = [
    #            total_points_0[[start_index, *total_disjoints_1[disjoint_start:disjoint_stop], stop_index - 1]]
    #            for (start_index, stop_index), (disjoint_start, disjoint_stop) in zip(
    #                it.pairwise((0, *total_disjoints_0, n_points)),
    #                it.pairwise((0, *disjoint_indices_1, len(total_disjoints_1))),
    #                strict=True
    #            )
    #        ]
    #        inlay_points_list_1 = [
    #            total_points_1[[start_index, *total_disjoints_0[disjoint_start:disjoint_stop], stop_index - 1]]
    #            for (start_index, stop_index), (disjoint_start, disjoint_stop) in zip(
    #                it.pairwise((0, *total_disjoints_1, n_points)),
    #                it.pairwise((0, *disjoint_indices_0, len(total_disjoints_0))),
    #                strict=True
    #            )
    #        ]
    #        inlay_disjoints_0 = np.cumsum([len(inlay_points) for inlay_points in inlay_points_list_0])
    #        inlay_disjoints_1 = np.cumsum([len(inlay_points) for inlay_points in inlay_points_list_1])

    #        total_points_0 = np.concatenate((
    #            total_points_0,
    #            *inlay_points_list_0,
    #            *(
    #                inlay_points.mean(axis=0, keepdims=True).repeat(len(inlay_points), axis=0)
    #                for inlay_points in inlay_points_list_1
    #            )
    #        ))
    #        total_points_1 = np.concatenate((
    #            total_points_1,
    #            *(
    #                inlay_points.mean(axis=0, keepdims=True).repeat(len(inlay_points), axis=0)
    #                for inlay_points in inlay_points_list_0
    #            ),
    #            *inlay_points_list_1
    #        ))
    #        total_disjoints = np.concatenate((
    #            total_disjoints,
    #            np.insert(inlay_disjoints_0[:-1], 0, 0) + n_points,
    #            np.insert(inlay_disjoints_1[:-1], 0, 0) + n_points + inlay_disjoints_0[-1]
    #        ))

    #    def callback(
    #        alpha: float
    #    ) -> Stroke:
    #        return Stroke(
    #            points=SpaceUtils.lerp(total_points_0, total_points_1)(alpha),
    #            disjoints=total_disjoints
    #        )

    #    return callback

    #@classmethod
    #def concatenate(
    #    cls,
    #    *graphs: "Graph"
    #) -> "Callable[[], Graph]":
    #    result = cls._concatenate(*graphs)

    #    def callback() -> Graph:
    #        return result

    #    return callback

    #@classmethod
    #def _concatenate(
    #    cls,
    #    *graphs: "Graph"
    #) -> "Graph":
    #    if not graphs:
    #        return Graph()

    #    positions = np.concatenate([
    #        graph._positions_
    #        for graph in graphs
    #    ])
    #    offsets = np.insert(np.cumsum([
    #        len(graph._positions_)
    #        for graph in graphs[:-1]
    #    ]), 0, 0)
    #    indices = np.concatenate([
    #        graph._indices_ + offset
    #        for graph, offset in zip(graphs, offsets, strict=True)
    #    ])
    #    return Graph(
    #        positions=positions,
    #        indices=indices
    #    )

    #@classmethod
    #def args_from_vertex_batches(
    #    cls,
    #    positions_batches: Iterable[NP_x3f8]
    #) -> tuple[NP_x3f8, NP_xi4]:
    #    batch_list = [
    #        batch for batch in positions_batches
    #        if len(batch) >= 2
    #    ]
    #    accumulated_lens = np.cumsum([len(batch) for batch in batch_list])
    #    concatenated_positions = SpaceUtils.increase_dimension(
    #        np.concatenate(batch_list) if batch_list else np.zeros((0, 3))
    #    )
    #    segment_indices = np.delete(np.arange(len(concatenated_positions)), accumulated_lens[:-1])[1:]
    #    return concatenated_positions, np.vstack((segment_indices - 1, segment_indices)).T

    #@classmethod
    #def from_vertex_batches(
    #    cls,
    #    positions_batches: Iterable[NP_x3f8]
    #) -> "Graph":
    #    positions, edges = cls.args_from_vertex_batches(positions_batches)
    #    return Graph(positions=positions, edges=edges)
