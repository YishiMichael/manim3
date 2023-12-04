from __future__ import annotations


import itertools
from typing import (
    Iterator,
    Literal,
    Never,
    Self,
    Unpack
)

import attrs
import numpy as np

from ..constants.custom_typing import (
    NP_x2i4,
    NP_x3f8,
    NP_xf8,
    NP_xi4
)
from ..lazy.lazy import Lazy
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
from .animatable.piecewisers import Piecewiser


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
        lengths = np.linalg.norm(positions[edges[:, 1]] - positions[edges[:, 0]], axis=1)
        return np.insert(np.cumsum(lengths), 0, 0.0)

    def set(
        self: Self,
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> Self:
        self._positions_ = positions
        self._edges_ = edges
        return self

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

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicGraph[Self]:
        return DynamicGraph(self, **kwargs)

    interpolate = GraphActions.interpolate.build_action_descriptor()
    piecewise = GraphActions.piecewise.build_action_descriptor()


class DynamicGraph[GraphT: Graph](DynamicAnimatable[GraphT]):
    __slots__ = ()

    interpolate = GraphActions.interpolate.build_dynamic_action_descriptor()
    piecewise = GraphActions.piecewise.build_dynamic_action_descriptor()


@attrs.frozen(kw_only=True)
class GraphInterpolateInfo:
    positions_0: NP_x3f8
    positions_1: NP_x3f8
    edges: NP_x2i4


class GraphInterpolateAnimation[GraphT: Graph](AnimatableInterpolateAnimation[GraphT]):
    __slots__ = ()

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        src_0: GraphT,
        src_1: GraphT
    ) -> GraphInterpolateInfo:
        positions_0, positions_1, edges = GraphUtils.graph_interpolate(src_0, src_1)
        return GraphInterpolateInfo(
            positions_0=positions_0,
            positions_1=positions_1,
            edges=edges
        )

    def interpolate(
        self: Self,
        dst: GraphT,
        alpha: float
    ) -> None:
        interpolate_info = self._interpolate_info_
        dst.set(
            positions=(1.0 - alpha) * interpolate_info.positions_0 + alpha * interpolate_info.positions_1,
            edges=interpolate_info.edges
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
    def concatenate_graphs(
        cls: type[Self],
        graphs: Iterator[tuple[NP_x3f8, NP_x2i4]]
    ) -> tuple[NP_x3f8, NP_x2i4]:
        graph_tuple: tuple[tuple[NP_x3f8, NP_x2i4], ...] = (
            (np.zeros((0, 3)), np.zeros((0, 2), dtype=np.int32)),
            *graphs
        )
        offsets = np.roll(np.fromiter((
            len(positions) for positions, _ in graph_tuple
        ), dtype=np.int32).cumsum(), 1)
        return (
            np.concatenate(tuple(positions for positions, _ in graph_tuple)),
            np.concatenate(tuple(edges + offset for (_, edges), offset in zip(graph_tuple, offsets, strict=True)))
        )

    @classmethod
    def graph_split(
        cls: type[Self],
        graph: Graph,
        alphas: NP_xf8
    ) -> Iterator[tuple[NP_x3f8, NP_x2i4]]:
        positions = graph._positions_
        edges = graph._edges_
        cumlengths = graph._cumlengths_
        if not len(edges):
            extended_positions = positions
            piece_edges_tuple = tuple(edges for _ in range(len(alphas) + 1))
        else:
            interpolated_positions, insertion_indices = cls._get_interpolated_samples(
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
        for piece_edges in piece_edges_tuple:
            (simplified_positions,), simplified_edges = cls._unify_edges((extended_positions, piece_edges))
            yield simplified_positions, simplified_edges

    @classmethod
    def graph_concatenate(
        cls: type[Self],
        graphs: tuple[Graph, ...]
    ) -> tuple[NP_x3f8, NP_x2i4]:
        (simplified_positions,), simplified_edges = cls._unify_edges(cls.concatenate_graphs(
            (graph._positions_, graph._edges_)
            for graph in graphs
        ))
        return simplified_positions, simplified_edges

    @classmethod
    def graph_interpolate(
        cls: type[Self],
        graph_0: Graph,
        graph_1: Graph
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        return cls._unify_aligned_graph_pairs(
            *cls._iter_aligned_graph_pairs_for_graph_interpolation(
                positions_0=graph_0._positions_,
                edges_0=graph_0._edges_,
                cumlengths_0=graph_0._cumlengths_,
                positions_1=graph_1._positions_,
                edges_1=graph_1._edges_,
                cumlengths_1=graph_1._cumlengths_
            )
        )

    @classmethod
    def _iter_aligned_graph_pairs_for_graph_interpolation(
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
            case _, 0:
                yield (
                    (positions_0, edges_0),
                    cls._get_centroid_graph(positions_0, edges_0, edges_0)
                )
            case 0, _:
                yield (
                    cls._get_centroid_graph(positions_1, edges_1, edges_1),
                    (positions_1, edges_1)
                )
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

    @classmethod
    def _unify_aligned_graph_pairs(
        cls: type[Self],
        *aligned_graph_pairs: tuple[tuple[NP_x3f8, NP_x2i4], tuple[NP_x3f8, NP_x2i4]]
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        aligned_graph_0 = cls.concatenate_graphs(
            graph_0 for graph_0, _ in aligned_graph_pairs
        )
        aligned_graph_1 = cls.concatenate_graphs(
            graph_1 for _, graph_1 in aligned_graph_pairs
        )
        (positions_0, positions_1), edges = cls._unify_edges(aligned_graph_0, aligned_graph_1)
        return positions_0, positions_1, edges

    @classmethod
    def _unify_edges(
        cls: type[Self],
        *aligned_graphs: tuple[NP_x3f8, NP_x2i4]
    ) -> tuple[tuple[NP_x3f8, ...], NP_x2i4]:
        unique_indices_array, inverse = np.unique(
            np.array(tuple(
                edges.flatten()
                for _, edges in aligned_graphs
            )),
            axis=1,
            return_inverse=True
        )
        return (
            tuple(
                positions[unique_indices]
                for (positions, _), unique_indices in zip(aligned_graphs, unique_indices_array, strict=True)
            ),
            inverse.reshape((-1, 2))
        )

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

    @classmethod
    def _extend_graph_samples(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        knots: NP_xf8,
        alphas: NP_xf8,
        side: Literal["left", "right"]
    ) -> tuple[NP_x3f8, NP_x2i4]:
        interpolated_positions, insertion_indices = cls._get_interpolated_samples(
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
    def _get_interpolated_samples(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        knots: NP_xf8,
        alphas: NP_xf8,
        side: Literal["left", "right"] = "left"
    ) -> tuple[NP_x3f8, NP_xi4]:
        insertion_indices = np.searchsorted(knots[1:-1], alphas, side=side).astype(np.int32)
        residues = (alphas - knots[insertion_indices]) / np.maximum(knots[insertion_indices + 1] - knots[insertion_indices], 1e-8)
        interpolated_positions = (
            (1.0 - residues[:, None]) * positions[edges[insertion_indices, 0]]
            + residues[:, None] * positions[edges[insertion_indices, 1]]
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
