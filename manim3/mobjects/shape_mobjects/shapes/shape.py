from functools import reduce
import itertools as it
from typing import (
    Iterable,
    Iterator
)

from mapbox_earcut import triangulate_float64
import numpy as np
import shapely.geometry
import shapely.validation

from ....constants.custom_typing import (
    NP_2f8,
    NP_x2f8,
    NP_xi4
)
from ....lazy.lazy import (
    Lazy,
    LazyObject
)
from ....utils.iterables import IterUtils
from ....utils.space import SpaceUtils
from ...graph_mobjects.graphs.graph import Graph
#from .stroke import Stroke
#from .line_string import LineString
#from .multi_line_string import MultiLineString


class Shape(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        graph: Graph | None = None
    ) -> None:
        super().__init__()
        if graph is not None:
            self._graph_ = graph

    def __and__(
        self,
        other: "Shape"
    ) -> "Shape":
        return self.intersection(other)

    def __or__(
        self,
        other: "Shape"
    ) -> "Shape":
        return self.union(other)

    def __sub__(
        self,
        other: "Shape"
    ) -> "Shape":
        return self.difference(other)

    def __xor__(
        self,
        other: "Shape"
    ) -> "Shape":
        return self.symmetric_difference(other)

    @Lazy.variable
    @classmethod
    def _graph_(cls) -> Graph:
        return Graph()

    @Lazy.property_external
    @classmethod
    def _shapely_obj_(
        cls,
        graph: Graph
    ) -> shapely.geometry.base.BaseGeometry:

        #def get_shapely_component(
        #    points: NP_x2f8
        #) -> shapely.geometry.base.BaseGeometry | None:
        #    #points = SpaceUtils.decrease_dimension(line_string._points_)
        #    if len(points) < 3:
        #        return None
        #    #    return shapely.geometry.Point(points[0])
        #    #if len(points) == 2:
        #    #    return shapely.geometry.LineString(points)
        #    return shapely.validation.make_valid(shapely.geometry.Polygon(points))

        positions_2d = SpaceUtils.decrease_dimension(graph._positions_)
        indices = graph._indices_
        disjoints = np.flatnonzero(indices[1:-2:2] - indices[2:-1:2]) + 1
        return reduce(shapely.geometry.base.BaseGeometry.__xor__, (
            shapely.validation.make_valid(shapely.geometry.Polygon(
                np.append(
                    positions_2d[indices[2 * start : 2 * stop : 2]],
                    [positions_2d[indices[2 * stop - 1]]],
                    axis=0
                )
            ))
            for start, stop in it.pairwise((0, *disjoints, len(indices) // 2))
            if stop - start >= 2
        ), shapely.geometry.GeometryCollection())

    @Lazy.property_external
    @classmethod
    def _triangulation_(
        cls,
        shapely_obj: shapely.geometry.base.BaseGeometry
    ) -> tuple[NP_xi4, NP_x2f8]:

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
        ) -> tuple[NP_xi4, NP_x2f8]:
            ring_positions_list = [
                np.array(boundary.coords)
                for boundary in [polygon.exterior, *polygon.interiors]
            ]
            positions = np.concatenate(ring_positions_list)
            if not len(positions):
                return np.arange(0, dtype=np.uint32), np.zeros((0, 2))

            ring_ends = np.cumsum([len(ring_positions) for ring_positions in ring_positions_list], dtype=np.uint32)
            return triangulate_float64(positions, ring_ends).astype(np.int32), positions

        def concatenate_triangulations(
            triangulations: Iterable[tuple[NP_xi4, NP_x2f8]]
        ) -> tuple[NP_xi4, NP_x2f8]:
            index_iterator, positions_iterator = IterUtils.unzip_pairs(triangulations)
            positions_list = list(positions_iterator)
            if not positions_list:
                return np.zeros((0,), dtype=np.int32), np.zeros((0, 2))

            offsets = np.cumsum((0, *(len(positions) for positions in positions_list[:-1])))
            all_indices = np.concatenate([
                index + offset
                for index, offset in zip(index_iterator, offsets, strict=True)
            ])
            all_positions = np.concatenate(positions_list)
            return all_indices, all_positions

        return concatenate_triangulations(
            get_polygon_triangulation(polygon)
            for polygon in get_shapely_polygons(shapely_obj)
        )

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[tuple[NP_x2f8, bool]]
    ) -> "Shape":
        path_list = [
            (positions, is_ring)
            for positions, is_ring in paths
            if len(positions)
        ]
        if not path_list:
            return Shape()

        positions = SpaceUtils.increase_dimension(np.concatenate([
            positions for positions, _ in path_list
        ]))
        offsets = np.insert(np.cumsum([len(positions) for positions, _ in path_list[:-1]], dtype=np.int32), 0, 0)
        indices = np.concatenate([
            Graph._get_consecutive_indices(len(positions), is_ring=is_ring) + offset
            for (positions, is_ring), offset in zip(path_list, offsets, strict=True)
        ])
        return Shape(Graph(
            positions=positions,
            indices=indices
        ))

    #@classmethod
    #def from_stroke(
    #    cls,
    #    stroke: Stroke
    #) -> "Shape":
    #    result = Shape()
    #    result._stroke_ = stroke
    #    return result

    @classmethod
    def from_shapely_obj(
        cls,
        shapely_obj: shapely.geometry.base.BaseGeometry
    ) -> "Shape":

        def iter_paths_from_shapely_obj(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Iterator[tuple[NP_x2f8, bool]]:
            match shapely_obj:
                case shapely.geometry.Point() | shapely.geometry.LineString():
                    yield np.array(shapely_obj.coords), False
                case shapely.geometry.Polygon():
                    yield np.array(shapely_obj.exterior.coords[:-1]), True
                    for interior in shapely_obj.interiors:
                        yield np.array(interior.coords[:-1]), True
                case shapely.geometry.base.BaseMultipartGeometry():
                    for shapely_obj_component in shapely_obj.geoms:
                        yield from iter_paths_from_shapely_obj(shapely_obj_component)
                case _:
                    raise TypeError

        return Shape.from_paths(iter_paths_from_shapely_obj(shapely_obj))

    #@classmethod
    #def partial(
    #    cls,
    #    shape: "Shape"
    #) -> "Callable[[float, float], Shape]":
    #    graph_partial_callback = Graph.partial(shape._graph_)

    #    def callback(
    #        alpha_0: float,
    #        alpha_1: float
    #    ) -> Shape:
    #        return Shape(graph_partial_callback(alpha_0, alpha_1))

    #    return callback

    #@classmethod
    #def interpolate(
    #    cls,
    #    shape_0: "Shape",
    #    shape_1: "Shape"
    #) -> "Callable[[float], Shape]":  # TODO
    #    graph_interpolate_callback = Graph.interpolate(
    #        shape_0._graph_, shape_1._graph_
    #    )

    #    

    #    def callback(
    #        alpha: float
    #    ) -> Shape:
    #        return Shape.from_shapely_obj(Shape(graph_interpolate_callback(alpha))._shapely_obj_)

    #    return callback

    #@classmethod
    #def concatenate(
    #    cls,
    #    *shapes: "Shape"
    #) -> "Callable[[], Shape]":
    #    graph_concatenate_callback = Graph.concatenate(*(
    #        shape._graph_
    #        for shape in shapes
    #    ))

    #    def callback() -> Shape:
    #        return Shape(graph_concatenate_callback())

    #    return callback

    # operations ported from shapely

    @property
    def shapely_obj(self) -> shapely.geometry.base.BaseGeometry:
        return self._shapely_obj_

    @property
    def area(self) -> float:
        return self.shapely_obj.area

    def distance(
        self,
        other: "Shape"
    ) -> float:
        return self.shapely_obj.distance(other.shapely_obj)

    def hausdorff_distance(
        self,
        other: "Shape"
    ) -> float:
        return self.shapely_obj.hausdorff_distance(other.shapely_obj)

    @property
    def length(self) -> float:
        return self.shapely_obj.length

    @property
    def centroid(self) -> NP_2f8:
        return np.array(self.shapely_obj.centroid)

    @property
    def convex_hull(self) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.convex_hull)

    @property
    def envelope(self) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.envelope)

    def buffer(
        self,
        distance: float,
        quad_segs: int = 16,
        cap_style: str = "round",
        join_style: str = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False
    ) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.buffer(
            distance=distance,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided
        ))

    def intersection(
        self,
        other: "Shape"
    ) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.intersection(other.shapely_obj))

    def union(
        self,
        other: "Shape"
    ) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.union(other.shapely_obj))

    def difference(
        self,
        other: "Shape"
    ) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.difference(other.shapely_obj))

    def symmetric_difference(
        self,
        other: "Shape"
    ) -> "Shape":
        return Shape.from_shapely_obj(self.shapely_obj.symmetric_difference(other.shapely_obj))
