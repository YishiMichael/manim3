from __future__ import annotations


import itertools
from typing import (
    Iterator,
    Literal,
    Self,
    Unpack
)

import numpy as np

from ..constants.custom_typing import (
    NP_3f8,
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

    @classmethod
    def _get_interpolate_info(
        cls: type[Self],
        graph_0: Graph,
        graph_1: Graph
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        graph_positions_0 = graph_0._positions_
        graph_positions_1 = graph_1._positions_
        graph_edges_0 = graph_0._edges_
        graph_edges_1 = graph_1._edges_

        match len(graph_edges_0), len(graph_edges_1):
            case 0, 0:
                return (
                    np.zeros((0, 3)),
                    np.zeros((0, 3)),
                    np.zeros((0,), dtype=np.int32)
                )
            case _, 0:
                centroid_0 = cls._get_centroid(graph_positions_0, graph_edges_0)
                return (
                    graph_positions_0,
                    np.repeat(centroid_0[None], len(graph_positions_0), axis=0),
                    graph_edges_0
                )
            case 0, _:
                centroid_1 = cls._get_centroid(graph_positions_1, graph_edges_1)
                return (
                    np.repeat(centroid_1[None], len(graph_positions_1), axis=0),
                    graph_positions_1,
                    graph_edges_1
                )

        knots_0 = graph_0._cumlengths_ * graph_1._cumlengths_[-1]
        knots_1 = graph_1._cumlengths_ * graph_0._cumlengths_[-1]
        interpolated_positions_0, indices_0 = cls._get_new_samples(
            graph_positions=graph_positions_0,
            graph_edges=graph_edges_0,
            knots=knots_0,
            alphas=knots_1[1:-1],
            side="right"
        )
        extended_positions_0, extended_edges_0 = cls._compose_samples(
            graph_positions=graph_positions_0,
            graph_edges=graph_edges_0,
            interpolated_positions=interpolated_positions_0,
            indices=indices_0
        )
        interpolated_positions_1, indices_1 = cls._get_new_samples(
            graph_positions=graph_positions_1,
            graph_edges=graph_edges_1,
            knots=knots_1,
            alphas=knots_0[1:-1],
            side="left"
        )
        extended_positions_1, extended_edges_1 = cls._compose_samples(
            graph_positions=graph_positions_1,
            graph_edges=graph_edges_1,
            interpolated_positions=interpolated_positions_1,
            indices=indices_1
        )
        return cls._pack_interpolate_info(
            positions_0=extended_positions_0,
            edges_0=extended_edges_0,
            positions_1=extended_positions_1,
            edges_1=extended_edges_1
        )

    @classmethod
    def _get_centroid(
        cls: type[Self],
        graph_positions: NP_x3f8,
        graph_edges: NP_x2i4
    ) -> NP_3f8:
        samples = graph_positions[graph_edges.flatten()]
        return (np.max(samples, axis=0) + np.min(samples, axis=0)) / 2.0

    @classmethod
    def _pack_interpolate_info(
        cls: type[Self],
        positions_0: NP_x3f8,
        edges_0: NP_x2i4,
        positions_1: NP_x3f8,
        edges_1: NP_x2i4
    ) -> tuple[NP_x3f8, NP_x3f8, NP_x2i4]:
        (unique_indices_0, unique_indices_1), inverse = np.unique(
            np.array((edges_0.flatten(), edges_1.flatten())),
            axis=1,
            return_inverse=True
        )
        return (
            positions_0[unique_indices_0],
            positions_1[unique_indices_1],
            inverse.reshape((-1, 2))
        )

    @classmethod
    def _get_new_samples(
        cls: type[Self],
        graph_positions: NP_x3f8,
        graph_edges: NP_x2i4,
        knots: NP_xf8,
        alphas: NP_xf8,
        side: Literal["left", "right"] = "left"
    ) -> tuple[NP_x3f8, NP_xi4]:
        indices = np.searchsorted(knots[1:-1], alphas, side=side).astype(np.int32)
        residues = (alphas - knots[indices]) / np.maximum(knots[indices + 1] - knots[indices], 1e-8)
        interpolated_positions = SpaceUtils.lerp(
            graph_positions[graph_edges[indices, 0]],
            graph_positions[graph_edges[indices, 1]],
            residues[:, None]
        )
        return interpolated_positions, indices

    @classmethod
    def _compose_samples(
        cls: type[Self],
        graph_positions: NP_x3f8,
        graph_edges: NP_x2i4,
        interpolated_positions: NP_x3f8,
        indices: NP_xi4
    ) -> tuple[NP_x3f8, NP_x2i4]:
        interpolated_position_indices = np.arange(len(indices)) + len(graph_positions)
        return (
            np.concatenate((graph_positions, interpolated_positions)),
            np.column_stack((
                np.insert(graph_edges[:, 0], indices + 1, interpolated_position_indices),
                np.insert(graph_edges[:, 1], indices, interpolated_position_indices)
            ))
        )

    def as_parameters(
        self: Self,
        positions: NP_x3f8,
        edges: NP_x2i4
    ) -> Self:
        self._positions_ = positions
        self._edges_ = edges
        return self

    def as_empty(
        self: Self
    ) -> Self:
        return self.as_parameters(
            positions=np.zeros((0, 3)),
            edges=np.zeros((0, 2), dtype=np.int32)
        )

    def split(
        self: Self,
        dsts: tuple[Self, ...],
        alphas: NP_xf8
    ) -> Self:
        positions = self._positions_
        edges = self._edges_
        if not len(edges):
            for dst in dsts:
                dst.as_empty()
            return self

        cls = type(self)
        knots = self._cumlengths_
        interpolated_positions, indices = cls._get_new_samples(
            graph_positions=positions,
            graph_edges=edges,
            knots=knots,
            alphas=alphas * knots[-1]
        )
        extended_positions, extended_edges = cls._compose_samples(
            graph_positions=positions,
            graph_edges=edges,
            interpolated_positions=interpolated_positions,
            indices=indices
        )
        slice_indices = np.array((0, *(indices + np.arange(len(indices)) + 1), len(edges) + len(indices)))
        for dst, (start, stop) in zip(dsts, itertools.pairwise(slice_indices), strict=True):
            unique_indices, inverse = np.unique(
                extended_edges[start:stop].flatten(),
                return_inverse=True
            )
            dst.as_parameters(
                positions=extended_positions[unique_indices],
                edges=inverse.reshape((-1, 2))
            )
        return self

    def concatenate(
        self: Self,
        srcs: tuple[Self, ...]
    ) -> Self:
        if not srcs:
            return self.as_empty()

        offsets = np.insert(np.cumsum([
            len(graph._positions_)
            for graph in srcs[:-1]
        ], dtype=np.int32), 0, 0)
        return self.as_parameters(
            positions=np.concatenate([
                graph._positions_
                for graph in srcs
            ]),
            edges=np.concatenate([
                graph._edges_ + offset
                for graph, offset in zip(srcs, offsets, strict=True)
            ])
        )

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
        return Graph._get_interpolate_info(
            graph_0=src_0,
            graph_1=src_1
        )

    def interpolate(
        self: Self,
        dst: GraphT,
        alpha: float
    ) -> None:
        positions_0, positions_1, edges = self._interpolate_info_
        dst.as_parameters(
            positions=SpaceUtils.lerp(positions_0, positions_1, alpha),
            edges=edges
        )

    def becomes(
        self: Self,
        dst: GraphT,
        src: GraphT
    ) -> None:
        dst.as_parameters(
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
