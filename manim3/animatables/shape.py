from __future__ import annotations


import itertools
from typing import (
    Iterable,
    Iterator,
    Literal,
    Self,
    Unpack
)

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
from .graph import (
    Graph,
    GraphUtils
)


class ShapeActions(AnimatableActions):
    __slots__ = ()

    @Action.register()
    @classmethod
    def interpolate(
        cls: type[Self],
        dst: Shape,
        src_0: Shape,
        src_1: Shape
    ) -> Iterator[Animation]:
        yield ShapeInterpolateAnimation(dst, src_0, src_1)

    @Action.register()
    @classmethod
    def piecewise(
        cls: type[Self],
        dst: Shape,
        src: Shape,
        piecewiser: Piecewiser
    ) -> Iterator[Animation]:
        yield ShapePiecewiseAnimation(dst, src, piecewiser)


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
            positions=SpaceUtils.increase_dimension(coordinates),
            edges=np.column_stack((edge_starts, edge_stops))
        )

    @Lazy.property()
    @staticmethod
    def _triangulation_(
        coordinates: NP_x2f8,
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
            ring_coordinates_list: list[NP_x2f8] = [
                contour.polygon,
                *(hole.polygon for hole in contour.children)
            ]
            coordinates = np.concatenate(ring_coordinates_list)
            ring_ends = np.cumsum([
                len(ring_coordinates) for ring_coordinates in ring_coordinates_list
            ], dtype=np.uint32)
            return mapbox_earcut.triangulate_float64(coordinates, ring_ends).reshape((-1, 3)).astype(np.int32), coordinates

        def concatenate_triangulations(
            triangulations: Iterable[tuple[NP_x3i4, NP_x2f8]]
        ) -> tuple[NP_x3i4, NP_x2f8]:
            triangulations_list = list(triangulations)
            coordinates_list = [
                coordinates for _, coordinates in triangulations_list
            ]
            if not coordinates_list:
                return np.zeros((0,), dtype=np.int32), np.zeros((0, 2))

            offsets = np.insert(np.cumsum([
                len(coordinates) for coordinates in coordinates_list[:-1]
            ], dtype=np.int32), 0, 0)
            all_faces = np.concatenate([
                faces + offset
                for (faces, _), offset in zip(triangulations_list, offsets, strict=True)
            ])
            all_coordinates = np.concatenate(coordinates_list)
            return all_faces, all_coordinates

        clipper = pyclipr.Clipper()
        clipper.addPaths([
            coordinates[start:stop]
            for start, stop in itertools.pairwise(cumcounts)
        ], pyclipr.Subject)
        poly_tree_root = clipper.execute2(pyclipr.Union, pyclipr.EvenOdd)
        return concatenate_triangulations(
            get_contour_triangulation(contour)
            for contour in iter_contour_nodes(poly_tree_root.children)
        )

    #@classmethod
    #def _get_interpolate_info(
    #    cls: type[Self],
    #    shape_0: Shape,
    #    shape_1: Shape
    #) -> tuple[NP_x2f8, NP_x2f8, NP_xi4]:
    #    graph_0 = shape_0._graph_
    #    graph_1 = shape_1._graph_
    #    outline_coordinates_0, outline_coordinates_1, outline_edges = Graph._get_interpolate_info(
    #        graph_0=graph_0,
    #        graph_1=graph_1
    #    )

    #    shape_cumcounts_0 = shape_0._cumcounts_
    #    shape_cumcounts_1 = shape_1._cumcounts_
    #    graph_positions_0 = graph_0._positions_
    #    graph_positions_1 = graph_1._positions_
    #    graph_edges_0 = graph_0._edges_
    #    graph_edges_1 = graph_1._edges_
    #    knots_0 = graph_0._cumlengths_ * graph_1._cumlengths_[-1]
    #    knots_1 = graph_1._cumlengths_ * graph_0._cumlengths_[-1]

    #    if not len(graph_edges_0) or not len(graph_edges_1):
    #        position_indices, counts = cls._get_position_indices_and_counts_from_edges(outline_edges)
    #        return (
    #            SpaceUtils.decrease_dimension(outline_positions_0)[position_indices],
    #            SpaceUtils.decrease_dimension(outline_positions_1)[position_indices],
    #            counts
    #        )

    #    inlay_interpolated_positions_0, inlay_indices_0 = Graph._get_new_samples(
    #        graph_positions=graph_positions_0,
    #        graph_edges=graph_edges_0,
    #        knots=knots_0,
    #        alphas=knots_1[shape_cumcounts_1][1:-1],
    #        side="right"
    #    )
    #    inlay_extended_positions_0, inlay_extended_edges_0 = Graph._insert_samples(
    #        graph_positions=graph_positions_0,
    #        graph_edges=np.column_stack((
    #            graph_edges_0[shape_cumcounts_0[:-1], 0],
    #            graph_edges_0[shape_cumcounts_0[1:] - 1, 1]
    #        )),
    #        interpolated_positions=inlay_interpolated_positions_0,
    #        indices=np.searchsorted(
    #            shape_cumcounts_0[1:-1] - 1,
    #            inlay_indices_0,
    #            side="right"
    #        ).astype(np.int32)
    #    )
    #    inlay_interpolated_positions_1, inlay_indices_1 = Graph._get_new_samples(
    #        graph_positions=graph_positions_1,
    #        graph_edges=graph_edges_1,
    #        knots=knots_1,
    #        alphas=knots_0[shape_cumcounts_0][1:-1],
    #        side="left"
    #    )
    #    inlay_extended_positions_1, inlay_extended_edges_1 = Graph._insert_samples(
    #        graph_positions=graph_positions_1,
    #        graph_edges=np.column_stack((
    #            graph_edges_1[shape_cumcounts_1[:-1], 0],
    #            graph_edges_1[shape_cumcounts_1[1:] - 1, 1]
    #        )),
    #        interpolated_positions=inlay_interpolated_positions_1,
    #        indices=np.searchsorted(
    #            shape_cumcounts_1[1:-1] - 1,
    #            inlay_indices_1,
    #            side="left"
    #        ).astype(np.int32)
    #    )
    #    all_positions_0, all_positions_1, all_edges = Graph._pack_interpolate_info(
    #        positions_0=np.concatenate((
    #            outline_positions_0,
    #            inlay_extended_positions_0,
    #            Graph._get_centroid(graph_positions_0, graph_edges_0)[None]
    #        )),
    #        edges_0=np.concatenate((
    #            outline_edges,
    #            inlay_extended_edges_0 + len(outline_positions_0),
    #            np.zeros_like(inlay_extended_edges_1) + (len(outline_positions_0) + len(inlay_extended_positions_0))
    #        )),
    #        positions_1=np.concatenate((
    #            outline_positions_1,
    #            Graph._get_centroid(graph_positions_1, graph_edges_1)[None],
    #            inlay_extended_positions_1
    #        )),
    #        edges_1=np.concatenate((
    #            outline_edges,
    #            np.zeros_like(inlay_extended_edges_0) + len(outline_positions_1),
    #            inlay_extended_edges_1 + (len(outline_positions_1) + 1)
    #        ))
    #    )
    #    position_indices, counts = cls._get_position_indices_and_counts_from_edges(all_edges)
    #    return (
    #        SpaceUtils.decrease_dimension(all_positions_0)[position_indices],
    #        SpaceUtils.decrease_dimension(all_positions_1)[position_indices],
    #        counts
    #    )

    def set(
        self: Self,
        coordinates: NP_x2f8,
        counts: NP_xi4
    ) -> Self:
        self._coordinates_ = coordinates
        self._counts_ = counts
        return self

    #def as_empty(
    #    self: Self
    #) -> Self:
    #    return self.set(
    #        positions=np.zeros((0, 3)),
    #        counts=np.zeros((0,), dtype=np.int32)
    #    )

    #def as_graph(
    #    self: Self,
    #    graph: Graph
    #) -> Self:
    #    position_indices, counts = type(self)._get_position_indices_and_counts_from_edges(graph._edges_)
    #    return self.set(
    #        positions=SpaceUtils.decrease_dimension(graph._positions_)[position_indices],
    #        counts=counts
    #    )

    def as_paths(
        self: Self,
        paths: Iterable[NP_x2f8]
    ) -> Self:
        path_list = list(paths)
        if not path_list:
            return self.set(
                coordinates=np.zeros((0, 3)),
                counts=np.zeros((0,), dtype=np.int32)
            )
        return self.set(
            coordinates=np.concatenate(path_list),
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
                shape._coordinates_[start:stop]
                for start, stop in itertools.pairwise(shape._cumcounts_)
            ], path_type)
        path_list: list[NP_x2f8] = clipper.execute(clip_type, fill_type)
        return self.as_paths(path_list)

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
        #return self.as_graph(Graph().concatenate(tuple(src._graph_ for src in srcs)))

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

    def animate(
        self: Self,
        **kwargs: Unpack[AnimateKwargs]
    ) -> DynamicShape[Self]:
        return DynamicShape(self, **kwargs)

    interpolate = ShapeActions.interpolate.build_animatable_method_descriptor()
    piecewise = ShapeActions.piecewise.build_animatable_method_descriptor()


class DynamicShape[ShapeT: Shape](DynamicAnimatable[ShapeT]):
    __slots__ = ()

    interpolate = ShapeActions.interpolate.build_dynamic_animatable_method_descriptor()
    piecewise = ShapeActions.piecewise.build_dynamic_animatable_method_descriptor()


class ShapeInterpolateAnimation[ShapeT: Shape](AnimatableInterpolateAnimation[ShapeT]):
    __slots__ = ()

    @Lazy.property()
    @staticmethod
    def _interpolate_info_(
        src_0: ShapeT,
        src_1: ShapeT
    ) -> tuple[NP_x2f8, NP_x2f8, NP_xi4]:
        return ShapeUtils.shape_interpolate_info(
            shape_0=src_0,
            shape_1=src_1
        )

    def interpolate(
        self: Self,
        dst: ShapeT,
        alpha: float
    ) -> None:
        coordinates_0, coordinates_1, counts = self._interpolate_info_
        dst.set(
            coordinates=SpaceUtils.lerp(coordinates_0, coordinates_1, alpha),
            counts=counts
        )

    def becomes(
        self: Self,
        dst: ShapeT,
        src: ShapeT
    ) -> None:
        dst.set(
            coordinates=src._coordinates_,
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


class ShapeUtils(GraphUtils):
    __slots__ = ()

    @classmethod
    def shape_split(
        cls: type[Self],
        shape: Shape,
        alphas: NP_xf8
    ) -> Iterator[tuple[NP_x2f8, NP_xi4]]:
        for positions, edges in cls.graph_split(shape._graph_, alphas):
            (coordinates,), counts = cls._unified_graph_group_to_unified_shape_group((positions,), edges)
            yield coordinates, counts

    @classmethod
    def shape_concatenate(
        cls: type[Self],
        shapes: tuple[Shape, ...]
        #graph_edges_tuple: tuple[NP_x2i4, ...],
    ) -> tuple[NP_x2f8, NP_xi4]:
        positions, edges = cls.graph_concatenate(tuple(shape._graph_ for shape in shapes))
        (coordinates,), counts = cls._unified_graph_group_to_unified_shape_group((positions,), edges)
        return coordinates, counts

    @classmethod
    def shape_interpolate_info(
        cls: type[Self],
        shape_0: Shape,
        shape_1: Shape
    ) -> tuple[NP_x2f8, NP_x2f8, NP_xi4]:
        graph_0 = shape_0._graph_
        graph_1 = shape_1._graph_
        aligned_graph_groups = tuple(cls._iter_aligned_graph_groups_for_shape_interpolation(
            positions_0=graph_0._positions_,
            edges_0=graph_0._edges_,
            cumlengths_0=graph_0._cumlengths_,
            cumcounts_0=shape_0._cumcounts_,
            positions_1=graph_1._positions_,
            edges_1=graph_1._edges_,
            cumlengths_1=graph_1._cumlengths_,
            cumcounts_1=shape_1._cumcounts_
        ))
        if not aligned_graph_groups:
            zipped_graph_group_0, zipped_graph_group_1 = (), ()
        else:
            zipped_graph_group_0, zipped_graph_group_1 = zip(*aligned_graph_groups, strict=True)
        (positions_0, positions_1), edges = cls._pack_aligned_graph_groups(zipped_graph_group_0, zipped_graph_group_1)
        (coordinates_0, coordinates_1), counts = cls._unified_graph_group_to_unified_shape_group((positions_0, positions_1), edges)
        return coordinates_0, coordinates_1, counts

    @classmethod
    def _unified_graph_group_to_unified_shape_group(
        cls: type[Self],
        positions_tuple: tuple[NP_x3f8, ...],
        edges: NP_x2i4
    ) -> tuple[tuple[NP_x2f8, ...], NP_xi4]:
        position_indices, counts = cls._get_position_indices_and_counts_from_edges(edges)
        return (
            tuple(
                SpaceUtils.decrease_dimension(positions)[position_indices]
                for positions in positions_tuple
            ),
            counts
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

    @classmethod
    def _iter_aligned_graph_groups_for_shape_interpolation(
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
        yield from cls._iter_aligned_graph_groups_for_graph_interpolation(
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
        #inlay_interpolated_positions_0, inlay_insertion_indices_0 = cls._get_new_samples(
        #    positions=positions_0,
        #    edges=edges_0,
        #    knots=knots_0,
        #    alphas=knots_1[cumcounts_1][1:-1],
        #    side="right"
        #)
        #inlay_extended_positions_0, inlay_extended_edges_0 = cls._insert_samples(
        #    positions=positions_0,
        #    edges=np.column_stack((
        #        edges_0[cumcounts_0[:-1], 0],
        #        edges_0[cumcounts_0[1:] - 1, 1]
        #    )),
        #    interpolated_positions=inlay_interpolated_positions_0,
        #    insertion_indices=np.searchsorted(
        #        cumcounts_0[1:-1] - 1,
        #        inlay_insertion_indices_0,
        #        side="right"
        #    ).astype(np.int32)
        #)
        #inlay_interpolated_positions_1, inlay_insertion_indices_1 = cls._get_new_samples(
        #    positions=positions_1,
        #    edges=edges_1,
        #    knots=knots_1,
        #    alphas=knots_0[cumcounts_0][1:-1],
        #    side="left"
        #)
        #inlay_extended_positions_1, inlay_extended_edges_1 = cls._insert_samples(
        #    positions=positions_1,
        #    edges=np.column_stack((
        #        edges_1[cumcounts_1[:-1], 0],
        #        edges_1[cumcounts_1[1:] - 1, 1]
        #    )),
        #    interpolated_positions=inlay_interpolated_positions_1,
        #    insertion_indices=np.searchsorted(
        #        cumcounts_1[1:-1] - 1,
        #        inlay_insertion_indices_1,
        #        side="left"
        #    ).astype(np.int32)
        #)
        yield (
            (inlay_positions_0, inlay_edges_0),
            cls._get_centroid_graph(positions_1, edges_1, inlay_edges_0)
        )
        yield (
            cls._get_centroid_graph(positions_0, edges_0, inlay_edges_1),
            (inlay_positions_1, inlay_edges_1)
        )
        #all_positions_0, all_positions_1, all_edges = cls._pack_interpolate_info(
        #    positions_0=np.concatenate((
        #        outline_positions_0,
        #        inlay_extended_positions_0,
        #        Graph._get_centroid(positions_0, edges_0)[None]
        #    )),
        #    edges_0=np.concatenate((
        #        outline_edges,
        #        inlay_extended_edges_0 + len(outline_positions_0),
        #        np.zeros_like(inlay_extended_edges_1) + (len(outline_positions_0) + len(inlay_extended_positions_0))
        #    )),
        #    positions_1=np.concatenate((
        #        outline_positions_1,
        #        Graph._get_centroid(positions_1, edges_1)[None],
        #        inlay_extended_positions_1
        #    )),
        #    edges_1=np.concatenate((
        #        outline_edges,
        #        np.zeros_like(inlay_extended_edges_0) + len(outline_positions_1),
        #        inlay_extended_edges_1 + (len(outline_positions_1) + 1)
        #    ))
        #)

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
        inlay_interpolated_positions, inlay_insertion_indices = cls._get_new_samples(
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

