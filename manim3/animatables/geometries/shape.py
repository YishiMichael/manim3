import functools
import itertools
from typing import (
    Iterable,
    Iterator,
    TypeVar
)

import mapbox_earcut
import numpy as np
import shapely.geometry
import shapely.validation

from ...constants.custom_typing import (
    NP_2f8,
    NP_x2f8,
    NP_x2i4,
    NP_x3i4,
    NP_xf8,
    NP_xi4
)
from ...lazy.lazy import Lazy
from ...utils.space_utils import SpaceUtils
from ..animatable.leaf_animatable import (
    LeafAnimatable,
    LeafAnimatableInterpolateInfo
)
from .graph import Graph
#from ..mobject.mobject_attributes.mobject_attribute import (
#    InterpolateHandler,
#    MobjectAttribute
#)


_ShapeT = TypeVar("_ShapeT", bound="Shape")


class Shape(LeafAnimatable):
    __slots__ = ()

    def __init__(
        self,
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
        self: _ShapeT,
        other: _ShapeT
    ):
        return self.intersection(other)

    def __or__(
        self: _ShapeT,
        other: _ShapeT
    ):
        return self.union(other)

    def __sub__(
        self: _ShapeT,
        other: _ShapeT
    ):
        return self.difference(other)

    def __xor__(
        self: _ShapeT,
        other: _ShapeT
    ):
        return self.symmetric_difference(other)

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _positions_() -> NP_x2f8:
        return np.zeros((0, 2))

    @Lazy.variable(hasher=Lazy.array_hasher)
    @staticmethod
    def _counts_() -> NP_xi4:
        return np.zeros((0,), dtype=np.int32)

    @Lazy.property(hasher=Lazy.array_hasher)
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
    def _shapely_obj_(
        #graph: Graph
        positions: NP_x2f8,
        cumcounts: NP_xi4
    ) -> shapely.geometry.base.BaseGeometry:

        #def get_polygon_positions(
        #    positions: NP_x2f8,
        #    cumcounts: NP_xi4
        #) -> Iterator[NP_x2f8]:
        #    positions = SpaceUtils.decrease_dimension(graph._positions_)
        #    edges = graph._edges_
        #    if not len(edges):
        #        return
        #    disjoints = Graph._get_disjoints(edges=edges)
        #    for start, stop in it.pairwise((0, *(disjoints + 1), len(edges))):
        #        indices = edges[start:stop, 0]
        #        if indices[0] != (tail_index := edges[stop - 1, 1]):
        #            indices = np.append(indices, tail_index)
        #        yield positions[indices]

        return functools.reduce(shapely.geometry.base.BaseGeometry.__xor__, (
            shapely.validation.make_valid(shapely.geometry.Polygon(positions[start:stop]))
            for start, stop in itertools.pairwise(cumcounts)
            if stop - start >= 3
            #for polygon_positions in get_polygon_positions(positions, cumcounts)
            #if len(polygon_positions) >= 3
        ), shapely.geometry.GeometryCollection())

    @Lazy.property()
    @staticmethod
    def _triangulation_(
        shapely_obj: shapely.geometry.base.BaseGeometry
    ) -> tuple[NP_x3i4, NP_x2f8]:

        def get_shapely_polygons(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Iterator[shapely.geometry.Polygon]:
            match shapely_obj:
                case shapely.geometry.Point() | shapely.geometry.LineString():
                    pass
                case shapely.geometry.Polygon():
                    yield shapely_obj
                case shapely.geometry.base.BaseMultipartGeometry():
                    for child in shapely_obj.geoms:
                        yield from get_shapely_polygons(child)
                case _:
                    raise TypeError

        def get_polygon_triangulation(
            polygon: shapely.geometry.Polygon
        ) -> tuple[NP_x3i4, NP_x2f8]:
            ring_positions_list: list[NP_x2f8] = [
                np.fromiter(boundary.coords, dtype=np.dtype((np.float64, (2,))))
                for boundary in (polygon.exterior, *polygon.interiors)
            ]
            positions = np.concatenate(ring_positions_list)
            if not len(positions):
                return np.arange(0, dtype=np.uint32), np.zeros((0, 2))

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

        return concatenate_triangulations(
            get_polygon_triangulation(polygon)
            for polygon in get_shapely_polygons(shapely_obj)
        )

    #@classmethod
    #def _interpolate(
    #    cls,
    #    shape_0: _ShapeT,
    #    shape_1: _ShapeT
    #) -> "ShapeInterpolateHandler":
    #    #graph_interpolate_handler = Graph._interpolate(shape_0._graph_, shape_1._graph_)
    #    #position_indices, counts = cls._get_position_indices_and_counts_from_edges(graph_interpolate_handler._edges)
    #    # TODO
    #    positions_0, positions_1, edges = Graph._general_interpolate(
    #        graph_0=shape_0._graph_,
    #        graph_1=shape_1._graph_,
    #        disjoints_0=shape_0._cumcounts_,
    #        disjoints_1=shape_1._cumcounts_
    #    )
    #    position_indices, counts = cls._get_position_indices_and_counts_from_edges(edges)
    #    return ShapeInterpolateHandler(
    #        positions_0=SpaceUtils.decrease_dimension(positions_0)[position_indices],
    #        positions_1=SpaceUtils.decrease_dimension(positions_1)[position_indices],
    #        counts=counts
    #    )

    #@classmethod
    #def to_graph(
    #    cls,
    #    shape: _ShapeT
    #) -> Graph:
    #    disjoints = shape._disjoints_
    #    rings = shape._rings_
    #    if not disjoints:
    #        return Graph()
    #    return Graph(
    #        positions=SpaceUtils.increase_dimension(shape._positions_),
    #        edges=np.concatenate()
    #    )

    #@classmethod
    #def _concatenate(
    #    cls,
    #    shapes: "list[Shape]"
    #):
    #    return cls.set_from_graph(Graph._concatenate([
    #        shape._graph_ for shape in shapes
    #    ]))

    #@classmethod
    #def _split(
    #    cls,
    #    shape: _ShapeT,
    #    alphas: NP_xf8
    #) -> "list[Shape]":
    #    return [
    #        cls.set_from_graph(graph)
    #        for graph in Graph._split(shape._graph_, alphas)
    #    ]

    @classmethod
    def _interpolate(
        cls: type[_ShapeT],
        src_0: _ShapeT,
        src_1: _ShapeT
    ) -> "ShapeInterpolateInfo[_ShapeT]":
        return ShapeInterpolateInfo(src_0, src_1)

    #def _interpolate(
    #    self: _ShapeT,
    #    src_0: _ShapeT,
    #    src_1: _ShapeT
    #) -> Updater:
    #    #graph_positions_0, graph_positions_1, edges = Graph._general_interpolate(
    #    #    graph_0=src_0._graph_,
    #    #    graph_1=src_1._graph_,
    #    #    disjoints_0=np.insert(np.cumsum(src_0._counts_), 0, 0),
    #    #    disjoints_1=np.insert(np.cumsum(src_1._counts_), 0, 0)
    #    #)
    #    #position_indices, counts = cls._get_position_indices_and_counts_from_edges(edges)
    #    #positions_0 = SpaceUtils.decrease_dimension(graph_positions_0)[position_indices]
    #    #positions_1 = SpaceUtils.decrease_dimension(graph_positions_1)[position_indices]
    #    return ShapeInterpolateUpdater(self, src_0, src_1)

    @classmethod
    def _split(
        cls: type[_ShapeT],
        #dst_tuple: tuple[_ShapeT, ...],
        src: _ShapeT,
        alphas: NP_xf8
    ) -> tuple[_ShapeT, ...]:
        return tuple(cls.from_graph(graph) for graph in Graph._split(src._graph_, alphas))

    @classmethod
    def _concatenate(
        cls: type[_ShapeT],
        #dst: _ShapeT,
        src_tuple: tuple[_ShapeT, ...]
    ) -> _ShapeT:
        return cls.from_graph(Graph._concatenate(tuple(src._graph_ for src in src_tuple)))

    @classmethod
    def _get_position_indices_and_counts_from_edges(
        cls,
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

    #def set_from_parameters(
    #    self,
    #    positions: NP_x2f8,
    #    counts: NP_xi4
    #):
    #    self._positions_ = positions
    #    self._counts_ = counts
    #    return self

    #def set_from_shape(
    #    self,
    #    shape: _ShapeT
    #):
    #    self.set_from_parameters(
    #        positions=shape._positions_,
    #        counts=shape._counts_
    #    )
    #    return self

    @classmethod
    def from_graph(
        cls,
        graph: Graph
    ):
        position_indices, counts = cls._get_position_indices_and_counts_from_edges(graph._edges_)
        return cls(
            positions=SpaceUtils.decrease_dimension(graph._positions_)[position_indices],
            counts=counts
        )

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[NP_x2f8]
    ):
        #path_list = [
        #    (positions, ring)
        #    for positions, ring in paths
        #    if len(positions)
        #]
        #if not path_list:
        #    return Shape()
        #return Shape(
        #    positions=np.concatenate([
        #        positions for positions, _ in path_list
        #    ]),
        #    disjoints=np.insert(np.cumsum([
        #        len(positions) for positions, _ in path_list
        #    ], dtype=np.int32), 0, 0),
        #    rings=np.fromiter((
        #        ring for _, ring in path_list
        #    ), dtype=np.bool_)
        #)
        path_list = list(paths)
        #positions = SpaceUtils.increase_dimension(np.concatenate(path_list))
        #offsets = np.insert(np.cumsum([
        #    len(positions) for positions, _ in path_list[:-1]
        #], dtype=np.int32), 0, 0)
        #edges = np.concatenate([
        #    Graph._get_consecutive_edges(len(positions), is_ring=is_ring) + offset
        #    for (positions, is_ring), offset in zip(path_list, offsets, strict=True)
        #])
        return cls(
            positions=np.concatenate(path_list),
            counts=np.fromiter((len(path) for path in path_list), dtype=np.int32)
        )

    @classmethod
    def from_shapely_obj(
        cls,
        shapely_obj: shapely.geometry.base.BaseGeometry
    ):

        def iter_paths_from_shapely_obj(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Iterator[NP_x2f8]:
            positions_dtype = np.dtype((np.float64, (2,)))
            match shapely_obj:
                case shapely.geometry.Point() | shapely.geometry.LineString():
                    yield np.fromiter(shapely_obj.coords, dtype=positions_dtype)
                case shapely.geometry.Polygon():
                    yield np.fromiter(shapely_obj.exterior.coords[:-1], dtype=positions_dtype)
                    for interior in shapely_obj.interiors:
                        yield np.fromiter(interior.coords[:-1], dtype=positions_dtype)
                case shapely.geometry.base.BaseMultipartGeometry():
                    for shapely_obj_component in shapely_obj.geoms:
                        yield from iter_paths_from_shapely_obj(shapely_obj_component)
                case _:
                    raise TypeError

        return cls.from_paths(iter_paths_from_shapely_obj(shapely_obj))

    # operations ported from shapely

    @property
    def shapely_obj(self) -> shapely.geometry.base.BaseGeometry:
        return self._shapely_obj_

    @property
    def area(self) -> float:
        return self.shapely_obj.area

    def distance(
        self: _ShapeT,
        other: _ShapeT
    ) -> float:
        return self.shapely_obj.distance(other.shapely_obj)

    def hausdorff_distance(
        self: _ShapeT,
        other: _ShapeT
    ) -> float:
        return self.shapely_obj.hausdorff_distance(other.shapely_obj)

    @property
    def length(self) -> float:
        return self.shapely_obj.length

    @property
    def centroid(self) -> NP_2f8:
        return np.array(self.shapely_obj.centroid)

    @property
    def convex_hull(self):
        return type(self).from_shapely_obj(self.shapely_obj.convex_hull)

    @property
    def envelope(self):
        return type(self).from_shapely_obj(self.shapely_obj.envelope)

    def buffer(
        self,
        distance: float,
        quad_segs: int = 16,
        cap_style: str = "round",
        join_style: str = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False
    ):
        return type(self).from_shapely_obj(self.shapely_obj.buffer(
            distance=distance,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided
        ))

    def intersection(
        self: _ShapeT,
        other: _ShapeT
    ):
        return type(self).from_shapely_obj(self.shapely_obj.intersection(other.shapely_obj))

    def union(
        self: _ShapeT,
        other: _ShapeT
    ):
        return type(self).from_shapely_obj(self.shapely_obj.union(other.shapely_obj))

    def difference(
        self: _ShapeT,
        other: _ShapeT
    ):
        return type(self).from_shapely_obj(self.shapely_obj.difference(other.shapely_obj))

    def symmetric_difference(
        self: _ShapeT,
        other: _ShapeT
    ):
        return type(self).from_shapely_obj(self.shapely_obj.symmetric_difference(other.shapely_obj))


    #def _get_interpolate_updater(
    #    self: _ShapeT,
    #    shape_0: _ShapeT,
    #    shape_1: _ShapeT
    #) -> "ShapeInterpolateUpdater":
    #    return ShapeInterpolateUpdater(
    #        shape=self,
    #        shape_0=shape_0,
    #        shape_1=shape_1
    #    )

    #def partial(
    #    self,
    #    alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]]
    #) -> "ShapePartialUpdater":
    #    return ShapePartialUpdater(
    #        shape=self,
    #        original_shape=self._copy(),
    #        alpha_to_segments=alpha_to_segments
    #    )


class ShapeInterpolateInfo(LeafAnimatableInterpolateInfo[_ShapeT]):
    __slots__ = (
        "_positions_0",
        "_positions_1",
        "_counts"
    )

    def __init__(
        self,
        src_0: _ShapeT,
        src_1: _ShapeT
    ) -> None:
        super().__init__(src_0, src_1)
        positions_0, positions_1, edges = Graph._general_interpolate(
            graph_0=src_0._graph_,
            graph_1=src_1._graph_,
            disjoints_0=np.insert(np.cumsum(src_0._counts_), 0, 0),
            disjoints_1=np.insert(np.cumsum(src_1._counts_), 0, 0)
        )
        position_indices, counts = Shape._get_position_indices_and_counts_from_edges(edges)
        self._positions_0: NP_x2f8 = SpaceUtils.decrease_dimension(positions_0)[position_indices]
        self._positions_1: NP_x2f8 = SpaceUtils.decrease_dimension(positions_1)[position_indices]
        self._counts: NP_xi4 = counts

    def interpolate(
        self,
        shape: _ShapeT,
        alpha: float
    ) -> None:
        shape._positions_ = SpaceUtils.lerp(self._positions_0, self._positions_1, alpha)
        shape._counts_ = self._counts


#class ShapeInterpolateUpdater(Updater):
#    __slots__ = ("_shape",)

#    def __init__(
#        self,
#        shape: Shape,
#        shape_0: Shape,
#        shape_1: Shape
#    ) -> None:
#        super().__init__()
#        #positions_0, positions_1, edges = Graph._general_interpolate(
#        #    graph_0=shape_0._graph_,
#        #    graph_1=shape_1._graph_,
#        #    disjoints_0=shape_0._cumcounts_,
#        #    disjoints_1=shape_1._cumcounts_
#        #)
#        #position_indices, counts = Shape._get_position_indices_and_counts_from_edges(edges)
#        #return ShapeInterpolateHandler(
#        #    positions_0=SpaceUtils.decrease_dimension(positions_0)[position_indices],
#        #    positions_1=SpaceUtils.decrease_dimension(positions_1)[position_indices],
#        #    counts=counts
#        #)
#        #positions_0, positions_1, edges = Graph._general_interpolate(
#        #    graph_0=graph_0,
#        #    graph_1=graph_1,
#        #    disjoints_0=np.zeros((0,), dtype=np.int32),
#        #    disjoints_1=np.zeros((0,), dtype=np.int32)
#        #)
#        self._shape: Shape = shape
#        self._shape_0_ = shape_0._copy()
#        self._shape_1_ = shape_1._copy()
#        #self._positions_0_ = positions_0
#        #self._positions_1_ = positions_1
#        #self._counts_ = counts

#    @Lazy.variable()
#    @staticmethod
#    def _shape_0_() -> Shape:  # frozen, so requires copying
#        return NotImplemented

#    @Lazy.variable()
#    @staticmethod
#    def _shape_1_() -> Shape:
#        return NotImplemented

#    @Lazy.property()
#    @staticmethod
#    def _interpolate_info_(
#        shape_0: Shape,
#        shape_1: Shape
#    ) -> ShapeInterpolateInfo:
#        return ShapeInterpolateInfo(shape_0, shape_1)

#    def update(
#        self,
#        alpha: float
#    ) -> None:
#        super().update(alpha)
#        self._interpolate_info_.interpolate(self._shape, alpha)

#    def update_boundary(
#        self,
#        boundary: BoundaryT
#    ) -> None:
#        super().update_boundary(boundary)
#        self._shape._copy_lazy_content(self._shape_1_ if boundary else self._shape_0_)


##class ShapePartialUpdater(Updater[Shape]):
##    __slots__ = (
##        "_shape",
##        "_original_shape",
##        "_alpha_to_segments"
##    )

##    def __init__(
##        self,
##        shape: Shape,
##        original_shape: Shape,
##        alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]]
##    ) -> None:
##        super().__init__(shape)
##        self._original_shape: Shape = original_shape
##        self._alpha_to_segments: Callable[[float], tuple[NP_xf8, list[int]]] = alpha_to_segments

##    def update(
##        self,
##        alpha: float
##    ) -> None:
##        split_alphas, concatenate_indices = self._alpha_to_segments(alpha)
##        shapes = Shape._split(self._original_shape, split_alphas)
##        shape = Shape._concatenate([shapes[index] for index in concatenate_indices])
##        Shape._copy_lazy_content(self._instance, shape)
##        #mobjects = [equivalent_cls() for _ in range(len(split_alphas) + 1)]
##        #equivalent_cls._split_into(
##        #    dst_mobject_list=mobjects,
##        #    src_mobject=original_mobject,
##        #    alphas=split_alphas
##        #)
##        #equivalent_cls._concatenate_into(
##        #    dst_mobject=mobject,
##        #    src_mobject_list=[mobjects[index] for index in concatenate_indices]
##        #)


##class ShapeInterpolateHandler(InterpolateHandler[Shape]):
##    __slots__ = (
##        "_positions_0",
##        "_positions_1",
##        "_counts"
##    )

##    def __init__(
##        self,
##        positions_0: NP_x2f8,
##        positions_1: NP_x2f8,
##        counts: NP_xi4
##    ) -> None:
##        super().__init__()
##        self._positions_0: NP_x2f8 = positions_0
##        self._positions_1: NP_x2f8 = positions_1
##        self._counts: NP_xi4 = counts

##    def _interpolate(
##        self,
##        alpha: float
##    ) -> Shape:
##        return Shape(
##            positions=SpaceUtils.lerp(self._positions_0, self._positions_1, alpha),
##            counts=self._counts
##        )
