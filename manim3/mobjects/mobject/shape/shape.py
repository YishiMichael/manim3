from functools import reduce
import itertools as it
from typing import (
    Callable,
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
from .stroke import Stroke
#from .line_string import LineString
#from .multi_line_string import MultiLineString


class Shape(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        stroke: Stroke | None = None
        #points_iterable: Iterable[NP_x2f8] | None = None
    ) -> None:
        super().__init__()
        if stroke is not None:
            self._stroke_ = stroke

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
    def _stroke_(cls) -> Stroke:
        return Stroke()

    @Lazy.property_external
    @classmethod
    def _shapely_obj_(
        cls,
        stroke: Stroke
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

        points = SpaceUtils.decrease_dimension(stroke._points_)
        return reduce(shapely.geometry.base.BaseGeometry.__xor__, (
            shapely.validation.make_valid(shapely.geometry.Polygon(points[start:stop]))
            for start, stop in it.pairwise((0, *stroke._disjoints_, len(points)))
            if stop - start >= 3
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
            ring_points_list = [
                np.array(boundary.coords)
                for boundary in [polygon.exterior, *polygon.interiors]
            ]
            points = np.concatenate(ring_points_list)
            if not len(points):
                return np.arange(0, dtype=np.uint32), np.zeros((0, 2))

            ring_ends = np.cumsum([len(ring_points) for ring_points in ring_points_list], dtype=np.uint32)
            return triangulate_float64(points, ring_ends).astype(np.int32), points

        def concatenate_triangulations(
            triangulations: Iterable[tuple[NP_xi4, NP_x2f8]]
        ) -> tuple[NP_xi4, NP_x2f8]:
            index_iterator, points_iterator = IterUtils.unzip_pairs(triangulations)
            points_list = list(points_iterator)
            if not points_list:
                return np.arange(0), np.zeros((0, 2))

            offsets = np.cumsum((0, *(len(points) for points in points_list[:-1])))
            all_index = np.concatenate([
                index + offset
                for index, offset in zip(index_iterator, offsets, strict=True)
            ])
            all_points = np.concatenate(points_list)
            return all_index, all_points

        return concatenate_triangulations(
            get_polygon_triangulation(polygon)
            for polygon in get_shapely_polygons(shapely_obj)
        )

    @classmethod
    def from_points_iterable(
        cls,
        points_iterable: Iterable[NP_x2f8]
    ) -> "Shape":
        points_list = [
            points for points in points_iterable
            if len(points) >= 2
        ]
        if not points_list:
            return Shape()
        return Shape(Stroke(
            points=SpaceUtils.increase_dimension(np.concatenate(points_list)),
            disjoints=np.cumsum([len(points) for points in points_list[:-1]])
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

        def iter_points_from_shapely_obj(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Iterator[NP_x2f8]:
            match shapely_obj:
                case shapely.geometry.Point() | shapely.geometry.LineString():
                    yield np.array(shapely_obj.coords)
                case shapely.geometry.Polygon():
                    yield np.array(shapely_obj.exterior.coords)
                    for interior in shapely_obj.interiors:
                        yield np.array(interior.coords)
                case shapely.geometry.base.BaseMultipartGeometry():
                    for shapely_obj_component in shapely_obj.geoms:
                        yield from iter_points_from_shapely_obj(shapely_obj_component)
                case _:
                    raise TypeError

        return Shape.from_points_iterable(iter_points_from_shapely_obj(shapely_obj))

    @classmethod
    def partial(
        cls,
        shape: "Shape"
    ) -> "Callable[[float, float], Shape]":
        stroke_partial_callback = Stroke.partial(shape._stroke_)

        def callback(
            start: float,
            stop: float
        ) -> Shape:
            return Shape(stroke_partial_callback(start, stop))

        return callback

    @classmethod
    def interpolate(
        cls,
        shape_0: "Shape",
        shape_1: "Shape"
    ) -> "Callable[[float], Shape]":
        stroke_interpolate_callback = Stroke._interpolate(
            shape_0._stroke_, shape_1._stroke_, has_inlay=True
        )

        def callback(
            alpha: float
        ) -> Shape:
            return Shape(stroke_interpolate_callback(alpha))

        return callback

    @classmethod
    def concatenate(
        cls,
        *shapes: "Shape"
    ) -> "Callable[[], Shape]":
        stroke_concatenate_callback = Stroke.concatenate(*(
            shape._stroke_
            for shape in shapes
        ))

        def callback() -> Shape:
            return Shape(stroke_concatenate_callback())

        return callback

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
