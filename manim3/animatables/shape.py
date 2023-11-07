from __future__ import annotations


import itertools
from typing import (
    Iterable,
    Iterator,
    Self,
    Unpack
)

import mapbox_earcut
import numpy as np
import pyclipr

from ..constants.custom_typing import (
    NP_x2f8,
    NP_x2i4,
    NP_x3i4,
    NP_xf8,
    NP_xi4
)
from ..lazy.lazy import Lazy
from ..utils.space_utils import SpaceUtils
from .animatable.actions import ActionMeta
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
from .graph import Graph


class ShapeActions(AnimatableActions):
    __slots__ = ()

    @ActionMeta.register
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Shape,
        src_0: Shape,
        src_1: Shape
    ) -> Iterator[Animation]:
        yield ShapeInterpolateAnimation(dst, src_0, src_1)

    @ActionMeta.register
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Shape,
        src: Shape,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        yield ShapePiecewiseAnimation(dst, src, piecewiser)


class Shape(ShapeActions, Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        # Should satisfy `all(counts > 0)` and `len(positions) == sum(counts)`.
        positions: NP_x2f8 | None = None,
        counts: NP_xi4 | None = None
    ) -> None:
        super().__init__()
        if positions is not None:
            self._positions_ = positions
        if counts is not None:
            self._counts_ = counts

    def __and__(
        self: Self,
        other: Self
    ) -> Self:
        return self.intersection(other)

    def __or__(
        self: Self,
        other: Self
    ) -> Self:
        return self.union(other)

    def __sub__(
        self: Self,
        other: Self
    ) -> Self:
        return self.difference(other)

    def __xor__(
        self: Self,
        other: Self
    ) -> Self:
        return self.xor(other)

    @Lazy.variable()
    @staticmethod
    def _positions_() -> NP_x2f8:
        return np.zeros((0, 2))

    @Lazy.variable()
    @staticmethod
    def _counts_() -> NP_xi4:
        return np.zeros((0,), dtype=np.int32)

    @Lazy.property()
    @staticmethod
    def _cumcounts_(
        counts: NP_xi4
    ) -> NP_xi4:
        return np.insert(np.cumsum(counts), 0, 0)

    @Lazy.property()
    @staticmethod
    def _graph_(
        positions: NP_x2f8,
        cumcounts: NP_xi4
    ) -> Graph:
        edges_0 = np.arange(len(positions), dtype=np.int32)
        edges_1 = np.arange(len(positions), dtype=np.int32)
        edges_1[cumcounts[:-1]] = np.roll(edges_1[cumcounts[:-1]], 1)
        edges_1 = np.roll(edges_1, -1)
        return Graph(
            positions=SpaceUtils.increase_dimension(positions),
            edges=np.column_stack((edges_0, edges_1))
        )

    @Lazy.property()
    @staticmethod
    def _triangulation_(
        positions: NP_x2f8,
        cumcounts: NP_xi4
    ) -> tuple[NP_x3i4, NP_x2f8]:

        def iter_contour_nodes(
            poly_trees: list[pyclipr.PolyTreeD]
        ) -> Iterator[pyclipr.PolyTreeD]:
            # http://www.angusj.com/clipper2/Docs/Units/Clipper.Engine/Classes/PolyTreeD/_Body.htm
            for poly_tree in poly_trees:
                yield poly_tree
                for hole in poly_tree.children:
                    yield from iter_contour_nodes(hole.children)

        def get_contour_triangulation(
            contour: pyclipr.PolyTreeD
        ) -> tuple[NP_x3i4, NP_x2f8]:
            ring_positions_list: list[NP_x2f8] = [
                contour.polygon,
                *(hole.polygon for hole in contour.children)
            ]
            positions = np.concatenate(ring_positions_list)
            ring_ends = np.cumsum([
                len(ring_positions) for ring_positions in ring_positions_list
            ], dtype=np.uint32)
            return mapbox_earcut.triangulate_float64(positions, ring_ends).reshape((-1, 3)).astype(np.int32), positions

        def concatenate_triangulations(
            triangulations: Iterable[tuple[NP_x3i4, NP_x2f8]]
        ) -> tuple[NP_x3i4, NP_x2f8]:
            triangulations_list = list(triangulations)
            positions_list = [
                positions for _, positions in triangulations_list
            ]
            if not positions_list:
                return np.zeros((0,), dtype=np.int32), np.zeros((0, 2))

            offsets = np.insert(np.cumsum([
                len(positions) for positions in positions_list[:-1]
            ], dtype=np.int32), 0, 0)
            all_faces = np.concatenate([
                faces + offset
                for (faces, _), offset in zip(triangulations_list, offsets, strict=True)
            ])
            all_positions = np.concatenate(positions_list)
            return all_faces, all_positions

        clipper = pyclipr.Clipper()
        clipper.addPaths([
            positions[start:stop]
            for start, stop in itertools.pairwise(cumcounts)
        ], pyclipr.Subject)
        poly_tree_root = clipper.execute2(pyclipr.Union, pyclipr.EvenOdd)
        return concatenate_triangulations(
            get_contour_triangulation(contour)
            for contour in iter_contour_nodes(poly_tree_root.children)
        )

    @classmethod
    def _get_position_indices_and_counts_from_edges(
        cls: type[Self],
        edges: NP_x2i4
    ) -> tuple[NP_xi4, NP_xi4]:
        if not len(edges):
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

        disjoints = np.insert(np.array((0, len(edges))), 1, np.flatnonzero(edges[:-1, 1] - edges[1:, 0]) + 1)
        not_ring_indices = np.flatnonzero(edges[disjoints[:-1], 0] - edges[disjoints[1:] - 1, 1])
        position_indices = np.insert(edges[:, 0], disjoints[not_ring_indices + 1], edges[disjoints[not_ring_indices + 1] - 1, 1])
        counts = np.diff(disjoints)
        counts[not_ring_indices] += 1
        return position_indices, counts

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicShape[Self]:
        return DynamicShape(self, **kwargs)

    def as_parameters(
        self: Self,
        positions: NP_x2f8,
        counts: NP_xi4
    ) -> Self:
        self._positions_ = positions
        self._counts_ = counts
        return self

    def as_empty(
        self: Self
    ) -> Self:
        return self.as_parameters(
            positions=np.zeros((0, 3)),
            counts=np.zeros((0,), dtype=np.int32)
        )

    def as_graph(
        self: Self,
        graph: Graph
    ) -> Self:
        position_indices, counts = type(self)._get_position_indices_and_counts_from_edges(graph._edges_)
        return self.as_parameters(
            positions=SpaceUtils.decrease_dimension(graph._positions_)[position_indices],
            counts=counts
        )

    def as_paths(
        self: Self,
        paths: Iterable[NP_x2f8]
    ) -> Self:
        path_list = list(paths)
        if not path_list:
            return self.as_empty()
        return self.as_parameters(
            positions=np.concatenate(path_list),
            counts=np.fromiter((len(path) for path in path_list), dtype=np.int32)
        )

    def as_clipping(
        self: Self,
        *shape_path_type_pairs: tuple[Self, pyclipr.PathType],
        # http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/ClipType.htm
        clip_type: pyclipr.ClipType,
        # http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/FillRule.htm
        fill_type: pyclipr.FillType
    ) -> Self:
        clipper = pyclipr.Clipper()
        for shape, path_type in shape_path_type_pairs:
            clipper.addPaths([
                shape._positions_[start:stop]
                for start, stop in itertools.pairwise(shape._cumcounts_)
            ], path_type)
        path_list: list[NP_x2f8] = clipper.execute(clip_type, fill_type)
        return self.as_paths(path_list)

    def split(
        self: Self,
        dsts: tuple[Self, ...],
        alphas: NP_xf8
    ) -> Self:
        graphs = tuple(Graph() for _ in dsts)
        self._graph_.split(graphs, alphas)
        for dst, graph in zip(dsts, graphs, strict=True):
            dst.as_graph(graph)
        return self

    def concatenate(
        self: Self,
        srcs: tuple[Self, ...]
    ) -> Self:
        return self.as_graph(Graph().concatenate(tuple(src._graph_ for src in srcs)))

    def intersection(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Intersection,
            fill_type=pyclipr.NonZero
        )

    def union(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Union,
            fill_type=pyclipr.NonZero
        )

    def difference(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Difference,
            fill_type=pyclipr.NonZero
        )

    def xor(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Xor,
            fill_type=pyclipr.NonZero
        )


class DynamicShape[ShapeT: Shape](ShapeActions, DynamicAnimatable[ShapeT]):
    __slots__ = ()


class ShapeInterpolateAnimation[ShapeT: Shape](AnimatableInterpolateAnimation[ShapeT]):
    __slots__ = ()

    def __init__(
        self: Self,
        dst: ShapeT,
        src_0: ShapeT,
        src_1: ShapeT
    ) -> None:
        super().__init__(dst, src_0, src_1)
        self._shape_0_ = src_0.copy()
        self._shape_1_ = src_1.copy()

    @Lazy.variable()
    @staticmethod
    def _shape_0_() -> ShapeT:
        return NotImplemented

    @Lazy.variable()
    @staticmethod
    def _shape_1_() -> ShapeT:
        return NotImplemented

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        shape_0: ShapeT,
        shape_1: ShapeT
    ) -> tuple[NP_x2f8, NP_x2f8, NP_xi4]:
        positions_0, positions_1, edges = Graph._general_interpolate(
            graph_0=shape_0._graph_,
            graph_1=shape_1._graph_,
            disjoints_0=np.insert(np.cumsum(shape_0._counts_), 0, 0),
            disjoints_1=np.insert(np.cumsum(shape_1._counts_), 0, 0)
        )
        position_indices, counts = Shape._get_position_indices_and_counts_from_edges(edges)
        return (
            SpaceUtils.decrease_dimension(positions_0)[position_indices],
            SpaceUtils.decrease_dimension(positions_1)[position_indices],
            counts
        )

    def interpolate(
        self: Self,
        dst: ShapeT,
        alpha: float
    ) -> None:
        positions_0, positions_1, counts = self._interpolate_info_
        dst.as_parameters(
            positions=SpaceUtils.lerp(positions_0, positions_1, alpha),
            counts=counts
        )

    def becomes(
        self: Self,
        dst: ShapeT,
        src: ShapeT
    ) -> None:
        dst.as_parameters(
            positions=src._positions_,
            counts=src._counts_
        )


class ShapePiecewiseAnimation[ShapeT: Shape](AnimatablePiecewiseAnimation[ShapeT]):
    __slots__ = ()

    @classmethod
    def split(
        cls: type[Self],
        dsts: tuple[ShapeT, ...],
        src: ShapeT,
        alphas: NP_xf8
    ) -> None:
        src.split(dsts, alphas)

    @classmethod
    def concatenate(
        cls: type[Self],
        dst: ShapeT,
        srcs: tuple[ShapeT, ...]
    ) -> None:
        dst.concatenate(srcs)
