__all__ = [
    "LineStringKind",
    "MultiLineString",
    "Shape"
]


from abc import abstractmethod
from enum import Enum
from functools import reduce
import itertools as it
from typing import (
    Callable,
    Generator,
    Iterable,
    Literal
)

from mapbox_earcut import triangulate_float32
import numpy as np
import shapely.geometry
import shapely.validation

from ..custom_typing import (
    FloatsT,
    Vec2T,
    Vec3T,
    Vec2sT,
    Vec3sT,
    VertexIndexType
)
from ..lazy.core import LazyObject
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..utils.space import SpaceUtils


class LineStringKind(Enum):
    POINT = 0
    LINE_STRING = 1
    LINEAR_RING = 2


class ShapeInterpolant(LazyObject):
    __slots__ = ()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _lengths_(cls) -> FloatsT:
        # Make sure all entries are non-zero to avoid zero divisions
        return np.zeros((0, 1))

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _length_(
        cls,
        lengths: FloatsT
    ) -> float:
        return lengths.sum()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _length_knots_(
        cls,
        lengths: FloatsT
    ) -> FloatsT:
        if not len(lengths):
            return np.ones(1)
        unnormalized_knots = np.insert(lengths.cumsum(), 0, 0.0)
        # Ensure the last entry is always precisely 1.0
        return unnormalized_knots / unnormalized_knots[-1]

    @abstractmethod
    def interpolate_point(
        self,
        alpha: float
    ) -> Vec3T:
        pass

    def interpolate_points(
        self,
        alphas: Iterable[float]
    ) -> Vec3sT:
        return np.array([self.interpolate_point(alpha) for alpha in alphas])

    @classmethod
    def _get_residue(
        cls,
        target: float,
        array: FloatsT,
        index: int
    ) -> float:
        return (target - array[index]) / (array[index + 1] - array[index])

    def _integer_interpolate(
        self,
        target: float,
        *,
        side: Literal["left", "right"] = "right"
    ) -> tuple[int, float]:
        """
        Assumed that `array` has at least 2 elements and already sorted, and that `0 = array[0] <= target <= array[-1]`.
        Returns `(i, (target - array[i - 1]) / (array[i] - array[i - 1]))` such that
        `1 <= i <= len(array) - 1` and `array[i - 1] <= target <= array[i]`.
        """
        array = self._length_knots_.value
        assert len(array) >= 2
        index = int(np.searchsorted(array, target, side=side))
        if index == 0:
            return 1, 0.0
        if index == len(array):
            return len(array) - 1, 1.0
        return index, self._get_residue(target, array, index - 1)

    @classmethod
    def _zip_knots(
        cls,
        *knots_lists: FloatsT
    ) -> tuple[tuple[list[list[float]], ...], list[tuple[tuple[int, float, float], ...]]]:
        all_knots = np.concatenate(knots_lists)
        all_list_indices = np.repeat(np.arange(len(knots_lists)), [
            len(knots) for knots in knots_lists
        ])
        unique_knots, unique_inverse, unique_counts = np.unique(all_knots, return_inverse=True, return_counts=True)
        unique_inverse_argsorted = np.argsort(unique_inverse)
        list_indices_groups = [
            all_list_indices[unique_inverse_argsorted[slice(*span)]]
            for span in it.pairwise(np.cumsum(unique_counts))
        ]
        assert len(list_indices_groups)
        assert len(list_indices_groups[-1]) == len(knots_lists)
        knot_indices = np.zeros(len(knots_lists), dtype=np.int_)
        residue_list = [0.0 for _ in knots_lists]
        residue_list_tuple = tuple([0.0] for _ in knots_lists)
        residue_list_list_tuple: tuple[list[list[float]]] = tuple([] for _ in knots_lists)
        triplet_tuple_list: list[tuple[tuple[int, float, float], ...]] = []
        for knot, list_indices in zip(unique_knots[1:], list_indices_groups, strict=True):
            triplet_list: list[tuple[int, float, float]] = []
            for index in range(len(knots_lists)):
                if index in list_indices:
                    stop_residue = 1.0
                else:
                    stop_residue = cls._get_residue(knot, knots_lists[index], knot_indices[index])
                residue_list_tuple[index].append(stop_residue)
                triplet_list.append((knot_indices[index], residue_list[index], stop_residue))
                if index in list_indices:
                    residue_list_list_tuple[index].append(residue_list_tuple[index][:])
                    residue_list_tuple[index].clear()
                    residue_list_tuple[index].append(0.0)
                    next_residue = 0.0
                    knot_indices[index] += 1
                else:
                    next_residue = stop_residue
                residue_list[index] = next_residue
            triplet_tuple_list.append(tuple(triplet_list))
        return residue_list_list_tuple, triplet_tuple_list


class LineString(ShapeInterpolant):
    __slots__ = ()

    def __init__(
        self,
        coords: Vec3sT
    ) -> None:
        # TODO: shall we first remove redundant adjacent points?
        assert len(coords)
        super().__init__()
        self._coords_ = coords

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _coords_(cls) -> Vec3sT:
        return np.zeros((0, 3))

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _kind_(
        cls,
        coords: Vec3sT
    ) -> LineStringKind:
        if len(coords) == 1:
            return LineStringKind.POINT
        if np.isclose(SpaceUtils.norm(coords[-1] - coords[0]), 0.0):
            return LineStringKind.LINEAR_RING
        return LineStringKind.LINE_STRING

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _lengths_(
        cls,
        coords: Vec3sT
    ) -> FloatsT:
        return np.maximum(SpaceUtils.norm(coords[1:] - coords[:-1]), 1e-6)

    def interpolate_point(
        self,
        alpha: float
    ) -> Vec3T:
        coords = self._coords_.value
        if self._kind_.value == LineStringKind.POINT:
            return coords[0]
        index, residue = self._integer_interpolate(alpha)
        return SpaceUtils.lerp(coords[index - 1], coords[index], residue)

    def partial(
        self,
        start: float,
        stop: float
    ) -> "LineString":
        coords = self._coords_.value
        if self._kind_.value == LineStringKind.POINT:
            new_coords = [coords[0]]
        else:
            start_index, start_residue = self._integer_interpolate(start, side="right")
            stop_index, stop_residue = self._integer_interpolate(stop, side="left")
            if start_index == stop_index and start_residue == stop_residue:
                new_coords = [
                    SpaceUtils.lerp(coords[start_index - 1], coords[start_index], start_residue)
                ]
            else:
                new_coords = [
                    SpaceUtils.lerp(coords[start_index - 1], coords[start_index], start_residue),
                    *coords[start_index:stop_index],
                    SpaceUtils.lerp(coords[stop_index - 1], coords[stop_index], stop_residue)
                ]
        return LineString(np.array(new_coords))

    @classmethod
    def interpolate_shape_callback(
        cls,
        line_string_0: "LineString",
        line_string_1: "LineString"
    ) -> "Callable[[float], LineString]":
        all_knots = np.unique(np.concatenate((line_string_0._length_knots_.value, line_string_1._length_knots_.value)))
        point_callbacks: list[Callable[[float], Vec3T]] = [
            SpaceUtils.lerp_callback(line_string_0.interpolate_point(knot), line_string_1.interpolate_point(knot))
            for knot in all_knots
        ]

        def callback(
            alpha: float
        ) -> LineString:
            return LineString(np.array([
                point_callback(alpha)
                for point_callback in point_callbacks
            ]))

        return callback


class MultiLineString(ShapeInterpolant):
    __slots__ = ()

    def __init__(
        self,
        coords_iterable: Iterable[Vec3sT] | None = None
    ) -> None:
        super().__init__()
        if coords_iterable is not None:
            self._line_strings_.add(*(LineString(coords) for coords in coords_iterable))

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _line_strings_(cls) -> list[LineString]:
        return []

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _lengths_(
        cls,
        line_strings__length: list[float]
    ) -> FloatsT:
        return np.maximum(np.array(line_strings__length), 1e-6)

    def interpolate_point(
        self,
        alpha: float
    ) -> Vec3T:
        if not self._line_strings_:
            raise ValueError("Attempting to interpolate an empty MultiLineString")
        index, residue = self._integer_interpolate(alpha)
        return self._line_strings_[index - 1].interpolate_point(residue)

    def partial(
        self,
        start: float,
        stop: float
    ) -> "MultiLineString":
        result = MultiLineString()
        line_strings = self._line_strings_
        if not line_strings:
            return result

        start_index, start_residue = self._integer_interpolate(start, side="right")
        stop_index, stop_residue = self._integer_interpolate(stop, side="left")
        if start_index == stop_index:
            result._line_strings_.add(line_strings[start_index - 1].partial(start_residue, stop_residue))
        else:
            result._line_strings_.add(
                line_strings[start_index - 1].partial(start_residue, 1.0),
                *line_strings[start_index:stop_index - 1],
                line_strings[stop_index - 1].partial(0.0, stop_residue)
            )
        return result

    @classmethod
    def interpolate_shape_callback(
        cls,
        multi_line_string_0: "MultiLineString",
        multi_line_string_1: "MultiLineString",
        *,
        has_inlay: bool = False
    ) -> "Callable[[float], MultiLineString]":
        line_strings_0 = multi_line_string_0._line_strings_
        line_strings_1 = multi_line_string_1._line_strings_
        if not line_strings_0 or not line_strings_1:
            raise ValueError("Attempting to interpolate an empty MultiLineString")

        (residue_list_list_0, residue_list_list_1), triplet_tuple_list = multi_line_string_0._zip_knots(
            multi_line_string_0._length_knots_.value, multi_line_string_1._length_knots_.value
        )
        line_string_callbacks: list[Callable[[float], LineString]] = [
            LineString.interpolate_shape_callback(
                line_strings_0[index_0].partial(start_residue_0, stop_residue_0),
                line_strings_1[index_1].partial(start_residue_1, stop_residue_1)
            )
            for (index_0, start_residue_0, stop_residue_0), (index_1, start_residue_1, stop_residue_1) in triplet_tuple_list
        ]

        inlay_callbacks: list[Callable[[float], Vec3sT]] = []
        for index_0, residues in enumerate(residue_list_list_0):
            coords = line_strings_0[index_0].interpolate_points(residues)
            if len(coords) == 2:
                continue
            coords_center = np.average(coords, axis=0)
            inlay_callbacks.append(SpaceUtils.lerp_callback(coords, coords_center))

        for index_1, residues in enumerate(residue_list_list_1):
            coords = line_strings_1[index_1].interpolate_points(residues)
            if len(coords) == 2:
                continue
            coords_center = np.average(coords, axis=0)
            inlay_callbacks.append(SpaceUtils.lerp_callback(coords_center, coords))

        def callback(
            alpha: float
        ) -> MultiLineString:
            result = MultiLineString()
            result._line_strings_.add(*(
                line_string_callback(alpha)
                for line_string_callback in line_string_callbacks
            ))
            if has_inlay:
                result._line_strings_.add(*(
                    LineString(inlay_callback(alpha))
                    for inlay_callback in inlay_callbacks
                ))
            return result
        return callback

    @classmethod
    def concatenate(
        cls,
        multi_line_strings: "Iterable[MultiLineString]"
    ) -> "MultiLineString":
        result = MultiLineString()
        result._line_strings_.add(*it.chain(*(
            multi_line_string._line_strings_
            for multi_line_string in multi_line_strings
        )))
        return result


class Shape(LazyObject):
    __slots__ = ()

    def __init__(
        self,
        coords_iterable: Iterable[Vec2sT] | None = None
    ) -> None:
        super().__init__()
        if coords_iterable is not None:
            self._multi_line_string_ = MultiLineString(
                SpaceUtils.increase_dimension(coords)
                for coords in coords_iterable
                if len(coords)
            )

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

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString:
        return MultiLineString()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _shapely_obj_(
        cls,
        _multi_line_string__line_strings_: list[LineString]
    ) -> shapely.geometry.base.BaseGeometry:

        def get_shapely_component(
            line_string: LineString
        ) -> shapely.geometry.base.BaseGeometry:
            coords = line_string._coords_.value[:, :2]
            if line_string._kind_.value == LineStringKind.POINT:
                return shapely.geometry.Point(coords[0])
            if len(coords) == 2:
                return shapely.geometry.LineString(coords)
            return shapely.validation.make_valid(shapely.geometry.Polygon(coords))

        return reduce(shapely.geometry.base.BaseGeometry.__xor__, [
            get_shapely_component(line_string)
            for line_string in _multi_line_string__line_strings_
        ], shapely.geometry.GeometryCollection())

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _triangulation_(
        cls,
        shapely_obj: shapely.geometry.base.BaseGeometry
    ) -> tuple[VertexIndexType, Vec2sT]:

        def get_shapely_polygons(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Generator[shapely.geometry.Polygon, None, None]:
            if isinstance(shapely_obj, shapely.geometry.Point | shapely.geometry.LineString):
                pass
            elif isinstance(shapely_obj, shapely.geometry.Polygon):
                yield shapely_obj
            elif isinstance(shapely_obj, shapely.geometry.base.BaseMultipartGeometry):
                for child in shapely_obj.geoms:
                    yield from get_shapely_polygons(child)
            else:
                raise TypeError

        def get_polygon_triangulation(
            polygon: shapely.geometry.Polygon
        ) -> tuple[VertexIndexType, Vec2sT]:
            ring_coords_list = [
                np.array(boundary.coords, dtype=np.float32)
                for boundary in [polygon.exterior, *polygon.interiors]
            ]
            coords = np.concatenate(ring_coords_list)
            if not len(coords):
                return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2))

            ring_ends = np.cumsum([len(ring_coords) for ring_coords in ring_coords_list], dtype=np.uint32)
            return triangulate_float32(coords, ring_ends), coords

        item_list: list[tuple[VertexIndexType, Vec2sT]] = []
        coords_len = 0
        for polygon in get_shapely_polygons(shapely_obj):
            index, coords = get_polygon_triangulation(polygon)
            item_list.append((index + coords_len, coords))
            coords_len += len(coords)

        if not item_list:
            return np.zeros((0,), dtype=np.uint32), np.zeros((0, 2))

        index_list, coords_list = zip(*item_list)
        return np.concatenate(index_list), np.concatenate(coords_list)

    @classmethod
    def from_multi_line_string(
        cls,
        multi_line_string: MultiLineString
    ) -> "Shape":
        result = Shape()
        result._multi_line_string_ = multi_line_string
        return result

    @classmethod
    def from_shapely_obj(
        cls,
        shapely_obj: shapely.geometry.base.BaseGeometry
    ) -> "Shape":

        def iter_coords_from_shapely_obj(
            shapely_obj: shapely.geometry.base.BaseGeometry
        ) -> Generator[Vec2sT, None, None]:
            if isinstance(shapely_obj, shapely.geometry.Point | shapely.geometry.LineString):
                yield np.array(shapely_obj.coords)
            elif isinstance(shapely_obj, shapely.geometry.Polygon):
                shapely_obj = shapely.geometry.polygon.orient(shapely_obj)  # TODO: needed?
                yield np.array(shapely_obj.exterior.coords)
                for interior in shapely_obj.interiors:
                    yield np.array(interior.coords)
            elif isinstance(shapely_obj, shapely.geometry.base.BaseMultipartGeometry):
                for shapely_obj_component in shapely_obj.geoms:
                    yield from iter_coords_from_shapely_obj(shapely_obj_component)
            else:
                raise TypeError

        return Shape(iter_coords_from_shapely_obj(shapely_obj))

    def interpolate_point(
        self,
        alpha: float
    ) -> Vec2T:
        return self._multi_line_string_.interpolate_point(alpha)[:2]

    def partial(
        self,
        start: float,
        stop: float
    ) -> "Shape":
        return Shape.from_multi_line_string(self._multi_line_string_.partial(start, stop))

    @classmethod
    def interpolate_shape_callback(
        cls,
        shape_0: "Shape",
        shape_1: "Shape",
        *,
        has_inlay: bool = True
    ) -> "Callable[[float], Shape]":
        multi_line_string_callback = MultiLineString.interpolate_shape_callback(
            shape_0._multi_line_string_,
            shape_1._multi_line_string_,
            has_inlay=has_inlay
        )

        def callback(
            alpha: float
        ) -> Shape:
            return Shape.from_multi_line_string(multi_line_string_callback(alpha))

        return callback

    @classmethod
    def concatenate(
        cls,
        shapes: "Iterable[Shape]"
    ) -> "Shape":
        return Shape.from_multi_line_string(MultiLineString.concatenate(
            shape._multi_line_string_
            for shape in shapes
        ))

    # operations ported from shapely

    @property
    def shapely_obj(self) -> shapely.geometry.base.BaseGeometry:
        return self._shapely_obj_.value

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
    def centroid(self) -> Vec2T:
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
