from __future__ import annotations


import itertools
from typing import (
    Iterator,
    Literal,
    Never,
    Self,
    Unpack
)

import numpy as np

from ..constants.custom_typing import (
    NP_x2i4,
    NP_x3f8,
    NP_xf8,
    NP_xi4
)
from ..lazy.lazy import Lazy
from ..utils.space_utils import SpaceUtils
from .animatable.actions import Action
from .animatable.animatable import (
    Animatable,
    AnimatableActions,
    AnimatableInterpolateAnimation,
    AnimatablePiecewiseAnimation,
    DynamicAnimatable
)
from .animatable.animation import (
    AnimateKwargs,
    Animation
)
from .animatable.piecewiser import Piecewiser


class GraphActions(AnimatableActions):
    __slots__ = ()

    @Action.register()
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Graph,
        src_0: Graph,
        src_1: Graph
    ) -> Iterator[Animation]:
        yield GraphInterpolateAnimation(dst, src_0, src_1)

    @Action.register()
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Graph,
        src: Graph,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        yield GraphPiecewiseAnimation(dst, src, piecewiser)


class Graph(Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        positions: NP_x3f8 | None = None,
        edges: NP_x2i4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if edges is not None:
            self._edges_ = edges

    @Lazy.variable()
    @staticmethod
    def _positions_() -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable()
    @staticmethod
    def _edges_() -> NP_x2i4:
        return np.zeros((0, 2), dtype=np.int32)

    @Lazy.property()
    @staticmethod
    def _cumlengths_(
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> NP_xf8:
        lengths = SpaceUtils.norm(positions[edges[:, 1]] - positions[edges[:, 0]])
        return np.insert(np.cumsum(lengths), 0, 0.0)

    def set(
        self: Self,
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> Self:
        self._positions_ = positions
        self._edges_ = edges
        return self

    #def as_empty(
    #    self: Self
    #) -> Self:
    #    return self.set(
    #        positions=np.zeros((0, 3)),
    #        edges=np.zeros((0, 2), dtype=np.int32)
    #    )

    def split(
        self: Self,
        dsts: tuple[Self, ...],
        alphas: NP_xf8
    ) -> Self:
        for dst, (positions, edges) in zip(dsts, GraphUtils.graph_split(self, alphas), strict=True):
            dst.set(
                positions=positions,
                edges=edges
            )
        return self
        #positions = self._positions_
        #edges = self._edges_
        #if not len(edges):
        #    for dst in dsts:
        #        dst.as_empty()
        #    return self

        #cls = type(self)
        #knots = self._cumlengths_
        #interpolated_positions, indices = cls._get_new_samples(
        #    graph_positions=positions,
        #    graph_edges=edges,
        #    knots=knots,
        #    alphas=alphas * knots[-1]
        #)
        #extended_positions, extended_edges = cls._insert_samples(
        #    graph_positions=positions,
        #    graph_edges=edges,
        #    interpolated_positions=interpolated_positions,
        #    indices=indices
        #)
        #slice_indices = np.array((0, *(indices + np.arange(len(indices)) + 1), len(edges) + len(indices)))
        #for dst, (start, stop) in zip(dsts, itertools.pairwise(slice_indices), strict=True):
        #    unique_indices, inverse = np.unique(
        #        extended_edges[start:stop].flatten(),
        #        return_inverse=True
        #    )
        #    dst.set(
        #        positions=extended_positions[unique_indices],
        #        edges=inverse.reshape((-1, 2))
        #    )
        #return self

    def concatenate(
        self: Self,
        srcs: tuple[Self, ...]
    ) -> Self:
        positions, edges = GraphUtils.graph_concatenate(srcs)
        self.set(
            positions=positions,
            edges=edges
        )
        return self
        #if not srcs:
        #    return self.as_empty()

        #offsets = np.insert(np.cumsum([
        #    len(graph._positions_)
        #    for graph in srcs[:-1]
        #], dtype=np.int32), 0, 0)
        #return self.set(
        #    positions=np.concatenate([
        #        graph._positions_
        #        for graph in srcs
        #    ]),
        #    edges=np.concatenate([
        #        graph._edges_ + offset
        #        for graph, offset in zip(srcs, offsets, strict=True)
        #    ])
        #)

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicGraph[Self]:
        return DynamicGraph(self, **kwargs)

    interpolate = GraphActions.interpolate.build_animatable_method_descriptor()
    piecewise = GraphActions.piecewise.build_animatable_method_descriptor()


class DynamicGraph[GraphT: Graph](DynamicAnimatable[GraphT]):
    __slots__ = ()

    interpolate = GraphActions.interpolate.build_dynamic_animatable_method_descriptor()
    piecewise = GraphActions.piecewise.build_dynamic_animatable_method_descriptor()


class GraphInterpolateAnimation[GraphT: Graph](AnimatableInterpolateAnimation[GraphT]):
    __slots__ = ()

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        src_0: GraphT,
        src_1: GraphT
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        return GraphUtils.graph_interpolate_info(src_0, src_1)

    def interpolate(
        self: Self,
        dst: GraphT,
        alpha: float
    ) -> None:
        positions_0, positions_1, edges = self._interpolate_info_
        dst.set(
            positions=SpaceUtils.lerp(positions_0, positions_1, alpha),
            edges=edges
        )

    def becomes(
        self: Self,
        dst: GraphT,
        src: GraphT
    ) -> None:
        dst.set(
            positions=src._positions_,
            edges=src._edges_
        )


class GraphPiecewiseAnimation[GraphT: Graph](AnimatablePiecewiseAnimation[GraphT]):
    __slots__ = ()

    @classmethod
    def split(
        cls: type[Self],
        dsts: tuple[GraphT, ...],
        src: GraphT,
        alphas: NP_xf8
    ) -> None:
        src.split(dsts, alphas)

    @classmethod
    def concatenate(
        cls: type[Self],
        dst: GraphT,
        srcs: tuple[GraphT, ...]
    ) -> None:
        dst.concatenate(srcs)


class GraphUtils:
    __slots__ = ()

    def __new__(
        cls: type[Self]
    ) -> Never:
        raise TypeError

    @classmethod
    def graph_split(
        cls: type[Self],
        graph: Graph,
        alphas: NP_xf8
    ) -> Iterator[tuple[NP_x3f8, NP_x2i4]]:
        positions = graph._positions_
        edges = graph._edges_
        cumlengths = graph._cumlengths_
        #positions = self._positions_
        #edges = self._edges_
        if not len(edges):
            extended_positions = positions
            piece_edges_tuple = tuple(edges for _ in range(len(alphas) + 1))
            #for _ in range(len(alphas) + 1):
            #    yield (
            #        np.zeros((0, 3)),
            #        np.zeros((0, 2), dtype=np.int32)
            #    )
            #return
            #for dst in dsts:
            #    dst.as_empty()
            #return self

        #knots = self._cumlengths_
        else:
            interpolated_positions, insertion_indices = cls._get_new_samples(
                positions=positions,
                edges=edges,
                knots=cumlengths,
                alphas=alphas * cumlengths[-1]
            )
            extended_positions, extended_edges = cls._insert_samples(
                positions=positions,
                edges=edges,
                interpolated_positions=interpolated_positions,
                insertion_indices=insertion_indices
            )
            piece_edges_tuple = tuple(
                extended_edges[start:stop]
                for start, stop in itertools.pairwise(
                    np.array((0, *(insertion_indices + np.arange(len(alphas)) + 1), len(edges) + len(alphas)))
                )
            )
        #slice_indices = np.array((0, *(indices + np.arange(len(indices)) + 1), len(edges) + len(indices)))
        for piece_edges in piece_edges_tuple:
            (simplified_positions,), simplified_edges = cls._pack_aligned_graph_groups(((extended_positions, piece_edges),))
            yield simplified_positions, simplified_edges
            #unique_indices, inverse = np.unique(
            #    extended_edges[start:stop].flatten(),
            #    return_inverse=True
            #)
            #yield (
            #    extended_positions[unique_indices],
            #    inverse.reshape((-1, 2))
            #)

    @classmethod
    def graph_concatenate(
        cls: type[Self],
        graphs: tuple[Graph, ...]
        #edges_tuple: tuple[NP_x2i4, ...],
    ) -> tuple[NP_x3f8, NP_x2i4]:
        (positions,), edges = cls._pack_aligned_graph_groups(tuple(
            (graph._positions_, graph._edges_)
            for graph in graphs
        ))
        return positions, edges
        #if not graph_positions_tuple:
        #    return (
        #        np.zeros((0, 3)),
        #        np.zeros((0, 2), dtype=np.int32)
        #    )

        #offsets = np.insert(np.cumsum([
        #    len(graph_positions)
        #    for graph_positions in graph_positions_tuple
        #], dtype=np.int32), 0, 0)
        #return (
        #    np.concatenate(graph_positions_tuple),
        #    np.concatenate([
        #        edges + offset
        #        for edges, offset in zip(graph_edges_tuple, offsets, strict=True)
        #    ])
        #)

    @classmethod
    def graph_interpolate_info(
        cls: type[Self],
        graph_0: Graph,
        graph_1: Graph
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        aligned_graph_groups = tuple(cls._iter_aligned_graph_groups_for_graph_interpolation(
            positions_0=graph_0._positions_,
            edges_0=graph_0._edges_,
            cumlengths_0=graph_0._cumlengths_,
            positions_1=graph_1._positions_,
            edges_1=graph_1._edges_,
            cumlengths_1=graph_1._cumlengths_
        ))
        if not aligned_graph_groups:
            zipped_graph_group_0, zipped_graph_group_1 = (), ()
        else:
            zipped_graph_group_0, zipped_graph_group_1 = zip(*aligned_graph_groups, strict=True)
        (positions_0, positions_1), edges = cls._pack_aligned_graph_groups(zipped_graph_group_0, zipped_graph_group_1)
        return positions_0, positions_1, edges
        #positions_0 = graph_0._positions_
        #positions_1 = graph_1._positions_
        #edges_0 = graph_0._edges_
        #edges_1 = graph_1._edges_

        #match len(edges_0), len(edges_1):
        #    case 0, 0:
        #        return (
        #            np.zeros((0, 3)),
        #            np.zeros((0, 3)),
        #            np.zeros((0,), dtype=np.int32)
        #        )
        #    case _, 0:
        #        centroid_0 = cls._get_centroid(positions_0, edges_0)
        #        return (
        #            positions_0,
        #            np.repeat(centroid_0[None], len(positions_0), axis=0),
        #            edges_0
        #        )
        #    case 0, _:
        #        centroid_1 = cls._get_centroid(positions_1, edges_1)
        #        return (
        #            np.repeat(centroid_1[None], len(positions_1), axis=0),
        #            positions_1,
        #            edges_1
        #        )

        #knots_0 = graph_0._cumlengths_ * graph_1._cumlengths_[-1]
        #knots_1 = graph_1._cumlengths_ * graph_0._cumlengths_[-1]
        #interpolated_positions_0, indices_0 = cls._get_new_samples(
        #    graph_positions=positions_0,
        #    graph_edges=edges_0,
        #    knots=knots_0,
        #    alphas=knots_1[1:-1],
        #    side="right"
        #)
        #extended_positions_0, extended_edges_0 = cls._insert_samples(
        #    graph_positions=positions_0,
        #    graph_edges=edges_0,
        #    interpolated_positions=interpolated_positions_0,
        #    indices=indices_0
        #)
        #interpolated_positions_1, indices_1 = cls._get_new_samples(
        #    graph_positions=positions_1,
        #    graph_edges=edges_1,
        #    knots=knots_1,
        #    alphas=knots_0[1:-1],
        #    side="left"
        #)
        #extended_positions_1, extended_edges_1 = cls._insert_samples(
        #    graph_positions=positions_1,
        #    graph_edges=edges_1,
        #    interpolated_positions=interpolated_positions_1,
        #    indices=indices_1
        #)
        #return cls._pack_interpolate_info(
        #    positions_0=extended_positions_0,
        #    edges_0=extended_edges_0,
        #    positions_1=extended_positions_1,
        #    edges_1=extended_edges_1
        #)

    @classmethod
    def _iter_aligned_graph_groups_for_graph_interpolation(
        cls: type[Self],
        positions_0: NP_x3f8,
        edges_0: NP_x2i4,
        cumlengths_0: NP_xf8,
        positions_1: NP_x3f8,
        edges_1: NP_x2i4,
        cumlengths_1: NP_xf8
    ) -> Iterator[tuple[tuple[NP_x3f8, NP_x2i4], tuple[NP_x3f8, NP_x2i4]]]:
        match len(edges_0), len(edges_1):
            case 0, 0:
                pass
                #return
                #return (
                #    np.zeros((0, 3)),
                #    np.zeros((0, 3)),
                #    np.zeros((0,), dtype=np.int32)
                #)
            case _, 0:
                #centroid_0 = cls._get_centroid_graph(positions_0, edges_0)
                yield (
                    (positions_0, edges_0),
                    cls._get_centroid_graph(positions_0, edges_0, edges_0)
                )
                #return
                #return (
                #    positions_0,
                #    np.repeat(centroid_0[None], len(positions_0), axis=0),
                #    edges_0
                #)
            case 0, _:
                #centroid_1 = cls._get_centroid_graph(positions_1, edges_1)
                yield (
                    cls._get_centroid_graph(positions_1, edges_1, edges_1),
                    (positions_1, edges_1)
                )
                #return
                #return (
                #    np.repeat(centroid_1[None], len(positions_1), axis=0),
                #    positions_1,
                #    edges_1
                #)
            case _:
                knots_0 = cumlengths_0 * cumlengths_1[-1]
                knots_1 = cumlengths_1 * cumlengths_0[-1]
                yield (
                    cls._extend_graph_samples(
                        positions=positions_0,
                        edges=edges_0,
                        knots=knots_0,
                        alphas=knots_1[1:-1],
                        side="right"
                    ),
                    cls._extend_graph_samples(
                        positions=positions_1,
                        edges=edges_1,
                        knots=knots_1,
                        alphas=knots_0[1:-1],
                        side="left"
                    )
                )
        #interpolated_positions_0, insertion_indices_0 = cls._get_new_samples(
        #    positions=positions_0,
        #    edges=edges_0,
        #    knots=knots_0,
        #    alphas=knots_1[1:-1],
        #    side="right"
        #)
        #extended_positions_0, extended_edges_0 = cls._insert_samples(
        #    positions=positions_0,
        #    edges=edges_0,
        #    interpolated_positions=interpolated_positions_0,
        #    insertion_indices=insertion_indices_0
        #)
        #interpolated_positions_1, insertion_indices_1 = cls._get_new_samples(
        #    positions=positions_1,
        #    edges=edges_1,
        #    knots=knots_1,
        #    alphas=knots_0[1:-1],
        #    side="left"
        #)
        #extended_positions_1, extended_edges_1 = cls._insert_samples(
        #    positions=positions_1,
        #    edges=edges_1,
        #    interpolated_positions=interpolated_positions_1,
        #    insertion_indices=insertion_indices_1
        #)
        #yield (
        #    (extended_positions_0, extended_edges_0),
        #    (extended_positions_1, extended_edges_1)
        #)
        #return cls._pack_interpolate_info(
        #    positions_0=extended_positions_0,
        #    edges_0=extended_edges_0,
        #    positions_1=extended_positions_1,
        #    edges_1=extended_edges_1
        #)

    @classmethod
    def _pack_aligned_graph_groups(
        cls: type[Self],
        *zipped_graph_groups: tuple[tuple[NP_x3f8, NP_x2i4], ...]
    ) -> tuple[tuple[NP_x3f8, ...], NP_x2i4]:

        def unify_edges(
            aligned_graph_group: tuple[tuple[NP_x3f8, NP_x2i4], ...]
        ) -> tuple[tuple[NP_x3f8, ...], NP_x2i4, int]:
            unique_indices_array, inverse = np.unique(
                np.array(tuple(
                    edges.flatten()
                    for _, edges in aligned_graph_group
                )),
                axis=1,
                return_inverse=True
            )
            return (
                tuple(
                    positions[unique_indices]
                    for (positions, _), unique_indices in zip(aligned_graph_group, unique_indices_array, strict=True)
                ),
                inverse.reshape((-1, 2)),
                unique_indices_array.shape[1]
            )

        unified_graph_groups = tuple(
            #(
            #    tuple(positions for positions, _ in aligned_graph_group),
            #    np.unique(
            #        np.array(tuple(edges.flatten() for _, edges in aligned_graph_group)),
            #        axis=1,
            #        return_inverse=True
            #    )
            #)
            unify_edges(aligned_graph_group)
            for aligned_graph_group in zip(*zipped_graph_groups, strict=True)
        )
        if not unified_graph_groups:
            return (
                tuple(np.zeros((0, 3)) for _ in range(len(zipped_graph_groups))),
                np.zeros((0, 2), dtype=np.int32)
            )

        offsets = np.insert(np.cumsum([
            offset_increment
            for _, _, offset_increment in unified_graph_groups[:-1]
        ], dtype=np.int32), 0, 0)
        return (
            tuple(
                np.concatenate(zipped_positions_tuple)
                for zipped_positions_tuple in zip(*(
                    positions for positions, _, _ in unified_graph_groups
                ), strict=True)
            ),
            np.concatenate([
                edges + offset
                for (_, edges, _), offset in zip(unified_graph_groups, offsets, strict=True)
            ])
        )
        #return (
        #    np.concatenate(graph_positions_tuple),
        #    np.concatenate([
        #        graph_edges + offset
        #        for graph_edges, offset in zip(graph_edges_tuple, offsets, strict=True)
        #    ])
        #)
        #return (
        #    positions_0[unique_indices_0],
        #    positions_1[unique_indices_1],
        #    inverse.reshape((-1, 2))
        #)

    @classmethod
    def _get_centroid_graph(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        aligned_edges: NP_x2i4
    ) -> tuple[NP_x3f8, NP_x2i4]:
        samples = positions[edges.flatten()]
        centroid = (np.max(samples, axis=0) + np.min(samples, axis=0)) / 2.0
        return centroid[None], np.zeros_like(aligned_edges)

    #@classmethod
    #def _pack_interpolate_info(
    #    cls: type[Self],
    #    positions_0: NP_x3f8,
    #    edges_0: NP_x2i4,
    #    positions_1: NP_x3f8,
    #    edges_1: NP_x2i4
    #) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
    #    (unique_indices_0, unique_indices_1), inverse = np.unique(
    #        np.array((edges_0.flatten(), edges_1.flatten())),
    #        axis=1,
    #        return_inverse=True
    #    )
    #    return (
    #        positions_0[unique_indices_0],
    #        positions_1[unique_indices_1],
    #        inverse.reshape((-1, 2))
    #    )

    @classmethod
    def _extend_graph_samples(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        knots: NP_xf8,
        alphas: NP_xf8,
        side: Literal["left", "right"]
    ) -> tuple[NP_x3f8, NP_x2i4]:
        interpolated_positions, insertion_indices = cls._get_new_samples(
            positions=positions,
            edges=edges,
            knots=knots,
            alphas=alphas,
            side=side
        )
        return cls._insert_samples(
            positions=positions,
            edges=edges,
            interpolated_positions=interpolated_positions,
            insertion_indices=insertion_indices
        )

    @classmethod
    def _get_new_samples(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        knots: NP_xf8,
        alphas: NP_xf8,
        side: Literal["left", "right"] = "left"
    ) -> tuple[NP_x3f8, NP_xi4]:
        insertion_indices = np.searchsorted(knots[1:-1], alphas, side=side).astype(np.int32)
        residues = (alphas - knots[insertion_indices]) / np.maximum(knots[insertion_indices + 1] - knots[insertion_indices], 1e-8)
        interpolated_positions = SpaceUtils.lerp(
            positions[edges[insertion_indices, 0]],
            positions[edges[insertion_indices, 1]],
            residues[:, None]
        )
        return interpolated_positions, insertion_indices

    @classmethod
    def _insert_samples(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        interpolated_positions: NP_x3f8,
        insertion_indices: NP_xi4
    ) -> tuple[NP_x3f8, NP_x2i4]:
        interpolated_position_indices = np.arange(len(insertion_indices)) + len(positions)
        return (
            np.concatenate((positions, interpolated_positions)),
            np.column_stack((
                np.insert(edges[:, 0], insertion_indices + 1, interpolated_position_indices),
                np.insert(edges[:, 1], insertion_indices, interpolated_position_indices)
            ))
        )
