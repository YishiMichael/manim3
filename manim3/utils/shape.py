__all__ = ["Shape"]


from abc import abstractmethod
from functools import reduce
from typing import (
    Generator,
    Generic,
    TypeVar
)

from mapbox_earcut import triangulate_float32
import numpy as np
import shapely.geometry
import shapely.ops

from ..custom_typing import (
    FloatsT,
    Real,
    Vec2T,
    Vec2sT,
    VertexIndicesType
)
from ..utils.lazy import (
    LazyBase,
    lazy_property,
    lazy_property_initializer,
    lazy_property_initializer_writable
)


_ChildT = TypeVar("_ChildT", bound="ShapeInterpolantBase")


class ShapeInterpolantBase(LazyBase):
    @lazy_property
    @staticmethod
    def _lengths_() -> FloatsT:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _length_(lengths: FloatsT) -> float:
        return max(lengths.sum(), 1e-6)

    @lazy_property
    @staticmethod
    def _length_knots_(lengths: FloatsT, length: float) -> FloatsT:
        return lengths.cumsum() / length

    @abstractmethod
    def interpolate_point(self, alpha: Real) -> Vec2T:
        pass

    @abstractmethod
    def interpolate_shape(self, other: "ShapeInterpolantBase", alpha: Real) -> "ShapeInterpolantBase":
        pass

    @abstractmethod
    def partial(self, start: Real, end: Real) -> "ShapeInterpolantBase":
        pass

    @classmethod
    def _integer_interpolate(cls, array: FloatsT, target: Real) -> tuple[int, float]:
        """
        Assumed that `array` is already sorted, and that `0 <= array[0] <= target <= array[-1]`
        Returns `(i, (target - array[i - 1]) / (array[i] - array[i - 1]))` such that
        `0 <= i <= len(array) - 1` and `array[i - 1] <= target <= array[i]`,
        where we've interpreted `array[-1]` as 0.
        """
        index = int(np.searchsorted(array, target))
        if index == 0:
            try:
                return 0, target / array[0]
            except ZeroDivisionError:
                return 0, 0.0
        if index == len(array):
            return len(array) - 1, 1.0
        try:
            return index, (target - array[index - 1]) / (array[index] - array[index - 1])
        except ZeroDivisionError:
            return index, 0.0


class ShapeInterpolant(Generic[_ChildT], ShapeInterpolantBase):
    def __init__(self, children: list[_ChildT] | None = None):
        super().__init__()
        if children is not None:
            self._children_.extend(children)

    @lazy_property_initializer
    @staticmethod
    def _children_() -> list[_ChildT]:
        return []

    @lazy_property
    @staticmethod
    def _lengths_(children: list[_ChildT]) -> FloatsT:
        return np.array([child._length_ for child in children])

    def interpolate_point(self, alpha: Real) -> Vec2T:
        index, residue = self._integer_interpolate(self._length_knots_, alpha)
        return self._children_[index].interpolate_point(residue)

    def interpolate_shape(self, other: "ShapeInterpolant[_ChildT]", alpha: Real):
        children = self._children_
        knots_0 = self._length_knots_
        knots_1 = other._length_knots_
        current_knot = 0.0
        start_0 = 0.0
        start_1 = 0.0
        ptr_0 = 0
        ptr_1 = 0
        new_children: list[_ChildT] = []
        while ptr_0 < len(knots_0) and ptr_1 < len(knots_1):
            knot_0 = knots_0[ptr_0]
            knot_1 = knots_1[ptr_1]
            next_knot = min(knot_0, knot_1)
            end_0 = (next_knot - current_knot) / (knot_0 - current_knot)
            end_1 = (next_knot - current_knot) / (knot_1 - current_knot)
            child_0 = children[ptr_0].partial(start_0, end_0)
            child_1 = children[ptr_1].partial(start_1, end_1)
            new_children.append(child_0.interpolate_shape(child_1, alpha))

            if knot_0 == next_knot:
                start_0 = 0.0
                ptr_0 += 1
            else:
                start_0 = end_0
            if knot_1 == next_knot:
                start_1 = 0.0
                ptr_1 += 1
            else:
                start_1 = end_1
            current_knot = next_knot

        return self.__class__(new_children)

    def partial(self, start: Real, end: Real):
        children = self._children_
        knots = self._length_knots_
        start_index, start_residue = self._integer_interpolate(knots, start)
        end_index, end_residue = self._integer_interpolate(knots, end)
        if start_index == end_index:
            new_children = [children[start_index].partial(start_residue, end_residue)]
        else:
            new_children = [
                children[start_index].partial(start_residue, 1.0),
                *children[start_index + 1:end_index],
                children[end_index].partial(0.0, end_residue)
            ]
        return self.__class__(new_children)


class LineString(ShapeInterpolantBase):
    def __init__(self, coords: Vec2sT):
        super().__init__()
        self._coords_ = coords

    @lazy_property_initializer_writable
    @staticmethod
    def _coords_() -> Vec2sT:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _shapely_obj_(coords: Vec2sT) -> shapely.geometry.base.BaseGeometry:
        coords_len = len(coords)
        if not coords_len:
            return shapely.geometry.base.EmptyGeometry()
        elif coords_len == 1:
            return shapely.geometry.Point(coords[0])
        elif coords_len == 2:
            return shapely.geometry.LineString(coords)
        return shapely.geometry.Polygon(coords)

    @lazy_property
    @staticmethod
    def _lengths_(coords: Vec2sT) -> FloatsT:
        return np.linalg.norm(coords[1:] - coords[:-1], axis=1)

    @classmethod
    def _lerp(cls, vec_0: Vec2T, vec_1: Vec2T, alpha: Real) -> Vec2T:
        return (1.0 - alpha) * vec_0 + alpha * vec_1

    def interpolate_point(self, alpha: Real) -> Vec2T:
        index, residue = self._integer_interpolate(self._length_knots_, alpha)
        return self._lerp(self._coords_[index], self._coords_[index + 1], residue)

    def interpolate_shape(self, other: "LineString", alpha: Real) -> "LineString":
        all_knots = sorted({0.0, *self._length_knots_, *other._length_knots_})
        return LineString(np.array([
            self._lerp(self.interpolate_point(knot), other.interpolate_point(knot), alpha)
            for knot in all_knots
        ]))

    def partial(self, start: Real, end: Real) -> "LineString":
        coords = self._coords_
        knots = self._length_knots_
        start_index, start_residue = self._integer_interpolate(knots, start)
        end_index, end_residue = self._integer_interpolate(knots, end)
        if start_index == end_index:
            if start_residue == end_residue:
                new_coords = [
                    self._lerp(coords[start_index], coords[start_index + 1], start_residue)
                ]
            else:
                new_coords = [
                    self._lerp(coords[start_index], coords[start_index + 1], start_residue),
                    self._lerp(coords[start_index], coords[start_index + 1], end_residue)
                ]
        else:
            new_coords = [
                self._lerp(coords[start_index], coords[start_index + 1], start_residue),
                *coords[start_index + 1:end_index],
                self._lerp(coords[end_index], coords[end_index + 1], end_residue)
            ]
        return LineString(np.array(new_coords))


class Polygon(ShapeInterpolant[LineString]):
    @lazy_property
    @staticmethod
    def _triangulation_(children: list[LineString]) -> tuple[Vec2sT, VertexIndicesType]:
        ring_coords_list = [np.array(ring._coords_, dtype=np.float32) for ring in children]
        coords = np.concatenate(ring_coords_list)
        ring_ends = np.cumsum([len(ring_coords) for ring_coords in ring_coords_list], dtype=np.uint32)
        return (coords, triangulate_float32(coords, ring_ends))


class Polygons(ShapeInterpolant[Polygon]):
    @lazy_property
    @staticmethod
    def _triangulation_(children: list[Polygon]) -> tuple[Vec2sT, VertexIndicesType]:
        if not children:
            return np.zeros((0, 2)), np.zeros((0,), dtype=np.uint32)

        item_list: list[tuple[Vec2sT, VertexIndicesType]] = []
        coords_len = 0
        for child in children:
            coords, indices = child._triangulation_
            item_list.append((coords, indices + coords_len))
            coords_len += len(coords)
        coords_list, indices_list = zip(*item_list)
        return np.concatenate(coords_list), np.concatenate(indices_list)


class Shape(LazyBase):
    def __init__(self, shapely_obj: shapely.geometry.base.BaseGeometry | None = None):
        super().__init__()
        if shapely_obj is not None:
            self._polygons_ = Polygons(list(self._polygons_from_shapely(shapely_obj)))

    @lazy_property_initializer_writable
    @staticmethod
    def _polygons_() -> Polygons:
        return Polygons()

    @classmethod
    def _polygons_from_shapely(cls, shapely_obj: shapely.geometry.base.BaseGeometry) -> Generator[Polygon, None, None]:
        if isinstance(shapely_obj, shapely.geometry.Point | shapely.geometry.LineString):
            yield Polygon([LineString(np.array(shapely_obj.coords))])
        elif isinstance(shapely_obj, shapely.geometry.Polygon):
            yield Polygon([
                LineString(np.array(boundary.coords))
                for boundary in [shapely_obj.exterior, *shapely_obj.interiors]
            ])
        elif isinstance(shapely_obj, shapely.geometry.base.BaseMultipartGeometry):
            for child in shapely_obj.geoms:
                yield from cls._polygons_from_shapely(child)
        else:
            raise TypeError

    def interpolate_point(self, alpha: Real) -> Vec2T:
        return self._polygons_.interpolate_point(alpha)

    def interpolate_shape(self, other: "Shape", alpha: Real) -> "Shape":
        return self._build_from_interpolation_result(
            self._polygons_.interpolate_shape(other._polygons_, alpha)
        )

    def partial(self, start: Real, end: Real) -> "Shape":
        return self._build_from_interpolation_result(
            self._polygons_.partial(start, end)
        )

    @lazy_property
    @staticmethod
    def _triangulation_(polygons: Polygons) -> tuple[Vec2sT, VertexIndicesType]:
        return polygons._triangulation_

    def _build_from_interpolation_result(self, polygons: Polygons) -> "Shape":
        return Shape(reduce(shapely.geometry.base.BaseGeometry.__xor__, [
            line_string._shapely_obj_
            for polygon in polygons._children_
            for line_string in polygon._children_
        ]))
