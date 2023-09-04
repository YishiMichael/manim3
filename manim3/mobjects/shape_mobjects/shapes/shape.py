import itertools as it
from functools import reduce
from typing import (
    Iterable,
    Iterator
)

import numpy as np
import shapely.geometry
import shapely.validation
from mapbox_earcut import triangulate_float64


from ....constants.custom_typing import (
    NP_2f8,
    NP_x2f8,
    NP_x3i4
)
from ....lazy.lazy import Lazy
from ....utils.space_utils import SpaceUtils
from ...graph_mobjects.graphs.graph import Graph
from ...mobject.mobject_attributes.mobject_attribute import MobjectAttribute


class Shape(MobjectAttribute):
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

    #@Lazy.variable_array
    #@classmethod
    #def _positions_(cls) -> NP_x2f8:
    #    return np.zeros((0, 2))

    #@Lazy.variable_array
    #@classmethod
    #def _counts_(cls) -> NP_xi4:
    #    return np.zeros((0,), dtype=np.int32)

    @Lazy.variable()
    @staticmethod
    def _graph_() -> Graph:
        return Graph()

    @Lazy.property()
    @staticmethod
    def _shapely_obj_(
        graph: Graph
        #positions: NP_x2f8,
        #counts: NP_xi4
    ) -> shapely.geometry.base.BaseGeometry:

        def get_polygon_positions(
            graph: Graph
        ) -> Iterator[NP_x2f8]:
            positions = SpaceUtils.decrease_dimension(graph._positions_)
            edges = graph._edges_
            if not len(edges):
                return
            disjoints = Graph._get_disjoints(edges=edges)
            for start, stop in it.pairwise((0, *(disjoints + 1), len(edges))):
                indices = edges[start:stop, 0]
                if indices[0] != (tail_index := edges[stop - 1, 1]):
                    indices = np.append(indices, tail_index)
                yield positions[indices]

        return reduce(shapely.geometry.base.BaseGeometry.__xor__, (
            shapely.validation.make_valid(shapely.geometry.Polygon(polygon_positions))
            #for start, stop in it.pairwise(disjoints)
            #if stop - start >= 3
            for polygon_positions in get_polygon_positions(graph)
            if len(polygon_positions) >= 3
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
            return triangulate_float64(positions, ring_ends).reshape((-1, 3)).astype(np.int32), positions

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
    #def to_graph(
    #    cls,
    #    shape: "Shape"
    #) -> Graph:
    #    disjoints = shape._disjoints_
    #    rings = shape._rings_
    #    if not disjoints:
    #        return Graph()
    #    return Graph(
    #        positions=SpaceUtils.increase_dimension(shape._positions_),
    #        edges=np.concatenate()
    #    )

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[tuple[NP_x2f8, bool]]
    ) -> "Shape":
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
        positions = SpaceUtils.increase_dimension(np.concatenate([
            positions for positions, _ in path_list
        ]))
        offsets = np.insert(np.cumsum([
            len(positions) for positions, _ in path_list[:-1]
        ], dtype=np.int32), 0, 0)
        edges = np.concatenate([
            Graph._get_consecutive_edges(len(positions), is_ring=is_ring) + offset
            for (positions, is_ring), offset in zip(path_list, offsets, strict=True)
        ])
        return Shape(Graph(
            positions=positions,
            edges=edges
        ))

    @classmethod
    def from_shapely_obj(
        cls,
        shapely_obj: shapely.geometry.base.BaseGeometry
    ) -> "Shape":

        def iter_paths_from_shapely_obj(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Iterator[tuple[NP_x2f8, bool]]:
            positions_dtype = np.dtype((np.float64, (2,)))
            match shapely_obj:
                case shapely.geometry.Point() | shapely.geometry.LineString():
                    yield np.fromiter(shapely_obj.coords, dtype=positions_dtype), False
                case shapely.geometry.Polygon():
                    yield np.fromiter(shapely_obj.exterior.coords[:-1], dtype=positions_dtype), True
                    for interior in shapely_obj.interiors:
                        yield np.fromiter(interior.coords[:-1], dtype=positions_dtype), True
                case shapely.geometry.base.BaseMultipartGeometry():
                    for shapely_obj_component in shapely_obj.geoms:
                        yield from iter_paths_from_shapely_obj(shapely_obj_component)
                case _:
                    raise TypeError

        return Shape.from_paths(iter_paths_from_shapely_obj(shapely_obj))

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
