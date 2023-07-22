from typing import (
    Callable,
    Iterable,
    Literal,
    overload
)

import numpy as np

from ....constants.custom_typing import (
    NP_f8,
    NP_i4,
    NP_x2i4,
    NP_x3f8,
    NP_xf8,
    NP_xi4
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
        vertices: NP_x3f8 | None = None,
        edges: NP_x2i4 | None = None
    ) -> None:
        super().__init__()
        if vertices is not None:
            self._vertices_ = vertices
        if edges is not None:
            self._edges_ = edges

    @Lazy.variable_array
    @classmethod
    def _vertices_(cls) -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable_array
    @classmethod
    def _edges_(cls) -> NP_x2i4:
        return np.zeros((0, 2), dtype=np.int32)

    @Lazy.property_array
    @classmethod
    def _knots_(
        cls,
        vertices: NP_x3f8,
        edges: NP_x2i4
    ) -> NP_xf8:
        lengths = SpaceUtils.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]])
        return np.insert(np.cumsum(lengths), 0, 0.0)

    @overload
    @classmethod
    def _interpolate_knots(
        cls,
        knots: NP_xf8,
        values: NP_f8,
        *,
        side: Literal["left", "right"]
    ) -> tuple[NP_i4, NP_f8]: ...

    @overload
    @classmethod
    def _interpolate_knots(
        cls,
        knots: NP_xf8,
        values: NP_xf8,
        *,
        side: Literal["left", "right"]
    ) -> tuple[NP_xi4, NP_xf8]: ...

    @classmethod
    def _interpolate_knots(
        cls,
        knots: NP_xf8,
        values: NP_f8 | NP_xf8,
        *,
        side: Literal["left", "right"]
    ) -> tuple[NP_i4, NP_f8] | tuple[NP_xi4, NP_xf8]:
        index = np.clip(np.searchsorted(knots, values, side=side), 1, len(knots) - 1, dtype=np.int32) - 1
        residue = (values - knots[index]) / np.maximum(knots[index + 1] - knots[index], 1e-6)
        return index, residue

    @classmethod
    def partial(
        cls,
        graph: "Graph"
    ) -> "Callable[[float, float], Graph]":
        vertices = graph._vertices_
        edges = graph._edges_
        knots = graph._knots_
        n_vertices = len(vertices)
        length = knots[-1]

        def callback(
            start: float,
            stop: float
        ) -> Graph:
            if start > stop:
                return Graph()
            start_index, start_residue = cls._interpolate_knots(knots, start * length, side="right")
            stop_index, stop_residue = cls._interpolate_knots(knots, stop * length, side="left")
            return Graph(
                vertices=np.concatenate((
                    vertices,
                    np.array((
                        SpaceUtils.lerp(*vertices[edges[start_index]])(start_residue),
                        SpaceUtils.lerp(*vertices[edges[stop_index]])(stop_residue)
                    ))
                )),
                edges=np.array((
                    np.array((n_vertices, edges[start_index][0])),
                    *edges[start_index + 1:stop_index],
                    np.array((edges[stop_index][1], n_vertices + 1))
                ))
            )

        return callback

    @classmethod
    def interpolate(
        cls,
        graph_0: "Graph",
        graph_1: "Graph"
    ) -> "Callable[[float], Graph]":
        vertices_0 = graph_0._vertices_
        vertices_1 = graph_1._vertices_
        edges_0 = graph_0._edges_
        edges_1 = graph_1._edges_
        assert len(edges_0)
        assert len(edges_1)

        knots_0 = graph_0._knots_ * graph_1._knots_[-1]
        knots_1 = graph_1._knots_ * graph_0._knots_[-1]
        indices_0, residues_0 = cls._interpolate_knots(knots_0, knots_1, side="right")
        indices_1, residues_1 = cls._interpolate_knots(knots_1, knots_0, side="left")
        interpolated_vertices_0 = SpaceUtils.lerp(vertices_0[indices_0 - 1], vertices_0[indices_0])(residues_0[:, None])
        interpolated_vertices_1 = SpaceUtils.lerp(vertices_1[indices_1 - 1], vertices_1[indices_1])(residues_1[:, None])

        edge_indices_0 = np.array((edges_0[:-1, 1], edges_0[1:, 0]))
        joint_booleans_0 = edge_indices_0[0] == edge_indices_0[1]
        preserved_vertex_indices_0 = np.delete(edge_indices_0.flatten(), 2 * joint_booleans_0.nonzero()[0])
        repeats_0 = np.where(joint_booleans_0, 1, 2)

        edge_indices_1 = np.array((edges_1[:-1, 1], edges_1[1:, 0]))
        joint_booleans_1 = edge_indices_1[0] == edge_indices_1[1]
        preserved_vertex_indices_1 = np.delete(edge_indices_1.flatten(), 2 * joint_booleans_1.nonzero()[0])
        repeats_1 = np.where(joint_booleans_1, 1, 2)

        # Merge together boundaries if they match.
        boundary_vertex_indices = np.array(((edges_0[0, 0], edges_1[0, 0]), (edges_0[-1, 1], edges_1[-1, 1])))
        ring_boolean = np.all(boundary_vertex_indices[0] == boundary_vertex_indices[1])
        # `preserved_boundary_indices.shape` is either `(2, 2)` or `(1, 2)`.
        preserved_boundary_indices = np.delete(boundary_vertex_indices, np.atleast_1d(ring_boolean).nonzero()[0], axis=0)

        extended_vertices_0 = np.concatenate((
            np.repeat(interpolated_vertices_0, repeats_1, axis=0),
            vertices_0[preserved_vertex_indices_0],
            vertices_0[preserved_boundary_indices[:, 0]]
        ))
        extended_vertices_1 = np.concatenate((
            vertices_1[preserved_vertex_indices_1],
            np.repeat(interpolated_vertices_1, repeats_0, axis=0),
            vertices_1[preserved_boundary_indices[:, 1]]
        ))
        concatenated_indices = np.concatenate((
            np.repeat(indices_0 + np.arange(len(indices_0)), repeats_1),
            np.repeat(indices_1 + np.arange(len(indices_1)), repeats_0)
        ))
        edges = np.array((
            len(concatenated_indices),
            *np.repeat(
                np.argsort(concatenated_indices),
                np.where(np.concatenate((
                    np.repeat(joint_booleans_1, repeats_1),
                    np.repeat(joint_booleans_0, repeats_0)
                ))[concatenated_indices], 2, 1)
            ),
            len(concatenated_indices) + len(preserved_boundary_indices) - 1
        )).reshape((-1, 2))

        def callback(
            alpha: float
        ) -> Graph:
            return Graph(
                vertices=SpaceUtils.lerp(extended_vertices_0, extended_vertices_1)(alpha),
                edges=edges
            )

        return callback

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

    @classmethod
    def concatenate(
        cls,
        *graphs: "Graph"
    ) -> "Callable[[], Graph]":
        result = cls._concatenate(*graphs)

        def callback() -> Graph:
            return result

        return callback

    @classmethod
    def _concatenate(
        cls,
        *graphs: "Graph"
    ) -> "Graph":
        if not graphs:
            return Graph()

        vertices = np.concatenate([
            graph._vertices_
            for graph in graphs
        ])
        offsets = np.insert(np.cumsum([
            len(graph._vertices_)
            for graph in graphs[:-1]
        ]), 0, 0)
        edges = np.concatenate([
            graph._edges_ + offset
            for graph, offset in zip(graphs, offsets, strict=True)
        ])
        return Graph(
            vertices=vertices,
            edges=edges
        )

    @classmethod
    def args_from_vertex_batches(
        cls,
        vertices_batches: Iterable[NP_x3f8]
    ) -> tuple[NP_x3f8, NP_x2i4]:
        batch_list = [
            batch for batch in vertices_batches
            if len(batch) >= 2
        ]
        accumulated_lens = np.cumsum([len(batch) for batch in batch_list])
        concatenated_vertices = SpaceUtils.increase_dimension(
            np.concatenate(batch_list) if batch_list else np.zeros((0, 3))
        )
        segment_indices = np.delete(np.arange(len(concatenated_vertices)), accumulated_lens[:-1])[1:]
        return concatenated_vertices, np.vstack((segment_indices - 1, segment_indices)).T

    @classmethod
    def from_vertex_batches(
        cls,
        vertices_batches: Iterable[NP_x3f8]
    ) -> "Graph":
        vertices, edges = cls.args_from_vertex_batches(vertices_batches)
        return Graph(vertices=vertices, edges=edges)
