from __future__ import annotations


import itertools
from typing import (
    Iterator,
    Literal,
    Self,
    Unpack
)

import attrs
import mapbox_earcut
import numpy as np
import pyclipr

from ..constants.custom_typing import (
    NP_x2f8,
    NP_x2i4,
    NP_x3f8,
    NP_x3i4,
    NP_xf8,
    NP_xi4
)
from ..lazy.lazy import Lazy
from .animatable.action import (
    DescriptiveAction,
    DescriptorParameters
)
from .animatable.animatable import (
    Animatable,
    AnimatableInterpolateAnimation,
    AnimatablePiecewiseAnimation,
    AnimatableTimeline
)
from .animatable.animation import (
    AnimateKwargs,
    Animation
)
from .animatable.piecewiser import Piecewiser
from .graph import (
    Graph,
    GraphUtils
)


@attrs.frozen(kw_only=True)
class Triangulation:
    coordinates: NP_x2f8
    faces: NP_x3i4


class Shape(Animatable):
    __slots__ = ()

    def __init__(
        self: Self,
        # Should satisfy `all(counts > 0)` and `len(coordinates) == sum(counts)`.
        coordinates: NP_x2f8 | None = None,
        counts: NP_xi4 | None = None
    ) -> None:
        super().__init__()
        if coordinates is not None:
            self._coordinates_ = coordinates
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
    def _coordinates_() -> NP_x2f8:
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
        coordinates: NP_x2f8,
        cumcounts: NP_xi4
    ) -> Graph:
        edge_starts = np.arange(len(coordinates), dtype=np.int32)
        edge_stops = np.arange(len(coordinates), dtype=np.int32)
        edge_stops[cumcounts[:-1]] = np.roll(edge_stops[cumcounts[:-1]], 1)
        edge_stops = np.roll(edge_stops, -1)
        return Graph(
            positions=np.concatenate((coordinates, np.zeros((len(coordinates), 1))), axis=1),
            edges=np.column_stack((edge_starts, edge_stops))
        )

    @Lazy.property()
    @staticmethod
    def _triangulation_(
        coordinates: NP_x2f8,
        cumcounts: NP_xi4
    ) -> Triangulation:

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
        ) -> tuple[NP_x2f8, NP_x3i4]:
            ring_coordinates_tuple: tuple[NP_x2f8, ...] = (
                contour.polygon,
                *(hole.polygon for hole in contour.children)
            )
            coordinates = np.concatenate(ring_coordinates_tuple)
            ring_ends = np.fromiter((
                len(ring_coordinates) for ring_coordinates in ring_coordinates_tuple
            ), dtype=np.uint32).cumsum()
            return (
                coordinates,
                mapbox_earcut.triangulate_float64(coordinates, ring_ends).reshape((-1, 3)).astype(np.int32)
            )

        clipper = pyclipr.Clipper()
        clipper.addPaths(tuple(
            coordinates[start:stop]
            for start, stop in itertools.pairwise(cumcounts)
        ), pyclipr.Subject)
        poly_tree_root = clipper.execute2(pyclipr.Union, pyclipr.FillRule.EvenOdd)
        triangulation_coordinates, triangulation_faces = ShapeUtils.concatenate_triangulations(
            get_contour_triangulation(contour)
            for contour in iter_contour_nodes(poly_tree_root.children)
        )
        return Triangulation(
            coordinates=triangulation_coordinates,
            faces=triangulation_faces
        )

    def set(
        self: Self,
        coordinates: NP_x2f8,
        counts: NP_xi4
    ) -> Self:
        self._coordinates_ = coordinates
        self._counts_ = counts
        return self

    def as_paths(
        self: Self,
        paths: Iterator[NP_x2f8]
    ) -> Self:
        cls = type(self)
        return self.concatenate(tuple(
            cls().set(
                coordinates=path,
                counts=np.array((len(path),), dtype=np.int32)
            )
            for path in paths
            if len(path)
        ))

    def as_clipping(
        self: Self,
        *shape_path_type_pairs: tuple[Self, pyclipr.PathType],
        # http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/ClipType.htm
        clip_type: pyclipr.ClipType,
        # http://www.angusj.com/clipper2/Docs/Units/Clipper/Types/FillRule.htm
        fillRule: pyclipr.FillRule
    ) -> Self:
        clipper = pyclipr.Clipper()
        for shape, path_type in shape_path_type_pairs:
            clipper.addPaths([
                shape._coordinates_[start:stop]
                for start, stop in itertools.pairwise(shape._cumcounts_)
            ], path_type)
        path_list: list[NP_x2f8] = clipper.execute(clip_type, fillRule)
        return self.as_paths(iter(path_list))

    def split(
        self: Self,
        dsts: tuple[Self, ...],
        alphas: NP_xf8
    ) -> Self:
        for dst, (coordinates, counts) in zip(dsts, ShapeUtils.shape_split(self, alphas), strict=True):
            dst.set(
                coordinates=coordinates,
                counts=counts
            )
        return self

    def concatenate(
        self: Self,
        srcs: tuple[Self, ...]
    ) -> Self:
        coordinates, counts = ShapeUtils.shape_concatenate(srcs)
        self.set(
            coordinates=coordinates,
            counts=counts
        )
        return self

    def intersection(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Intersection,
            fillRule=pyclipr.FillRule.NonZero
        )

    def union(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Union,
            fillRule=pyclipr.FillRule.NonZero
        )

    def difference(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Difference,
            fillRule=pyclipr.FillRule.NonZero
        )

    def xor(
        self: Self,
        other: Self
    ) -> Self:
        return type(self)().as_clipping(
            (self, pyclipr.Subject),
            (other, pyclipr.Clip),
            clip_type=pyclipr.Xor,
            fillRule=pyclipr.FillRule.NonZero
        )

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> ShapeTimeline[Self]:
        return ShapeTimeline(self, **kwargs)

    @DescriptiveAction.descriptive_register(DescriptorParameters)
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Self,
        src_0: Self,
        src_1: Self
    ) -> Iterator[Animation]:
        yield ShapeInterpolateAnimation(dst, src_0, src_1)

    @DescriptiveAction.descriptive_register(DescriptorParameters)
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Self,
        src: Self,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        yield ShapePiecewiseAnimation(dst, src, piecewiser)


class ShapeTimeline[ShapeT: Shape](AnimatableTimeline[ShapeT]):
    __slots__ = ()

    interpolate = Shape.interpolate
    piecewise = Shape.piecewise


@attrs.frozen(kw_only=True)
class ShapeInterpolateInfo:
    coordinates_0: NP_x2f8
    coordinates_1: NP_x2f8
    counts: NP_xi4


class ShapeInterpolateAnimation[ShapeT: Shape](AnimatableInterpolateAnimation[ShapeT]):
    __slots__ = ()

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        src_0: ShapeT,
        src_1: ShapeT
    ) -> ShapeInterpolateInfo:
        coordinates_0, coordinates_1, counts = ShapeUtils.shape_interpolate(
            shape_0=src_0,
            shape_1=src_1
        )
        return ShapeInterpolateInfo(
            coordinates_0=coordinates_0,
            coordinates_1=coordinates_1,
            counts=counts
        )

    def interpolate(
        self: Self,
        dst: ShapeT,
        alpha: float
    ) -> None:
        interpolate_info = self._interpolate_info_
        dst.set(
            coordinates=(1.0 - alpha) * interpolate_info.coordinates_0 + alpha * interpolate_info.coordinates_1,
            counts=interpolate_info.counts
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


class ShapeUtils(GraphUtils):
    __slots__ = ()

    @classmethod
    def concatenate_triangulations(
        cls: type[Self],
        triangulations: Iterator[tuple[NP_x2f8, NP_x3i4]]
    ) -> tuple[NP_x2f8, NP_x3i4]:
        triangulation_tuple: tuple[tuple[NP_x2f8, NP_x3i4], ...] = (
            (np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int32)),
            *triangulations
        )
        offsets = np.roll(np.fromiter((
            len(coordinates) for coordinates, _ in triangulation_tuple
        ), dtype=np.int32).cumsum(), 1)
        return (
            np.concatenate(tuple(coordinates for coordinates, _ in triangulation_tuple)),
            np.concatenate(tuple(faces + offset for (_, faces), offset in zip(triangulation_tuple, offsets, strict=True)))
        )

    @classmethod
    def shape_split(
        cls: type[Self],
        shape: Shape,
        alphas: NP_xf8
    ) -> Iterator[tuple[NP_x2f8, NP_xi4]]:
        for positions, edges in cls.graph_split(shape._graph_, alphas):
            (coordinates,), counts = cls._unified_graphs_to_unified_shapes((positions,), edges)
            yield coordinates, counts

    @classmethod
    def shape_concatenate(
        cls: type[Self],
        shapes: tuple[Shape, ...]
    ) -> tuple[NP_x2f8, NP_xi4]:
        positions, edges = cls.graph_concatenate(tuple(shape._graph_ for shape in shapes))
        (coordinates,), counts = cls._unified_graphs_to_unified_shapes((positions,), edges)
        return coordinates, counts

    @classmethod
    def shape_interpolate(
        cls: type[Self],
        shape_0: Shape,
        shape_1: Shape
    ) -> tuple[NP_x2f8, NP_x2f8, NP_xi4]:
        graph_0 = shape_0._graph_
        graph_1 = shape_1._graph_
        positions_0, positions_1, edges = cls._unify_aligned_graph_pairs(
            *cls._iter_aligned_graph_pairs_for_shape_interpolation(
                positions_0=graph_0._positions_,
                edges_0=graph_0._edges_,
                cumlengths_0=graph_0._cumlengths_,
                cumcounts_0=shape_0._cumcounts_,
                positions_1=graph_1._positions_,
                edges_1=graph_1._edges_,
                cumlengths_1=graph_1._cumlengths_,
                cumcounts_1=shape_1._cumcounts_
            )
        )
        (coordinates_0, coordinates_1), counts = cls._unified_graphs_to_unified_shapes((positions_0, positions_1), edges)
        return coordinates_0, coordinates_1, counts

    @classmethod
    def _unified_graphs_to_unified_shapes(
        cls: type[Self],
        positions_tuple: tuple[NP_x3f8, ...],
        edges: NP_x2i4
    ) -> tuple[tuple[NP_x2f8, ...], NP_xi4]:
        if not len(edges):
            indices = np.zeros((0,), dtype=np.int32)
            counts = np.zeros((0,), dtype=np.int32)
        else:
            disjoints = np.array((0, *np.flatnonzero(edges[:-1, 1] - edges[1:, 0]) + 1, len(edges)))
            open_path_indices = np.flatnonzero(edges[disjoints[:-1], 0] - edges[disjoints[1:] - 1, 1])
            indices = np.insert(edges[:, 0], disjoints[open_path_indices + 1], edges[disjoints[open_path_indices + 1] - 1, 1])
            counts = np.diff(disjoints)
            counts[open_path_indices] += 1
        return (
            tuple(
                positions[indices, :2]
                for positions in positions_tuple
            ),
            counts
        )

    @classmethod
    def _iter_aligned_graph_pairs_for_shape_interpolation(
        cls: type[Self],
        positions_0: NP_x3f8,
        edges_0: NP_x2i4,
        cumlengths_0: NP_xf8,
        cumcounts_0: NP_xi4,
        positions_1: NP_x3f8,
        edges_1: NP_x2i4,
        cumlengths_1: NP_xf8,
        cumcounts_1: NP_xi4
    ) -> Iterator[tuple[tuple[NP_x3f8, NP_x2i4], tuple[NP_x3f8, NP_x2i4]]]:
        yield from cls._iter_aligned_graph_pairs_for_graph_interpolation(
            positions_0=positions_0,
            edges_0=edges_0,
            cumlengths_0=cumlengths_0,
            positions_1=positions_1,
            edges_1=edges_1,
            cumlengths_1=cumlengths_1
        )

        if not len(edges_0) or not len(edges_1):
            return

        knots_0 = cumlengths_0 * cumlengths_1[-1]
        knots_1 = cumlengths_1 * cumlengths_0[-1]
        inlay_positions_0, inlay_edges_0 = cls._get_inlay_graph(
            positions=positions_0,
            edges=edges_0,
            knots=knots_0,
            cumcounts=cumcounts_0,
            alphas=knots_1[cumcounts_1][1:-1],
            side="right"
        )
        inlay_positions_1, inlay_edges_1 = cls._get_inlay_graph(
            positions=positions_1,
            edges=edges_1,
            knots=knots_1,
            cumcounts=cumcounts_1,
            alphas=knots_0[cumcounts_0][1:-1],
            side="left"
        )
        yield (
            (inlay_positions_0, inlay_edges_0),
            cls._get_centroid_graph(positions_1, edges_1, inlay_edges_0)
        )
        yield (
            cls._get_centroid_graph(positions_0, edges_0, inlay_edges_1),
            (inlay_positions_1, inlay_edges_1)
        )

    @classmethod
    def _get_inlay_graph(
        cls: type[Self],
        positions: NP_x3f8,
        edges: NP_x2i4,
        knots: NP_xf8,
        cumcounts: NP_xi4,
        alphas: NP_xf8,
        side: Literal["left", "right"]
    ) -> tuple[NP_x3f8, NP_x2i4]:
        inlay_interpolated_positions, inlay_insertion_indices = cls._get_interpolated_samples(
            positions=positions,
            edges=edges,
            knots=knots,
            alphas=alphas,
            side=side
        )
        return cls._insert_samples(
            positions=positions,
            edges=np.column_stack((
                edges[cumcounts[:-1], 0],
                edges[cumcounts[1:] - 1, 1]
            )),
            interpolated_positions=inlay_interpolated_positions,
            insertion_indices=np.searchsorted(
                cumcounts[1:-1] - 1,
                inlay_insertion_indices,
                side=side
            ).astype(np.int32)
        )

