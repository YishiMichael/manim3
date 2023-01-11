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
import scipy.interpolate
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


#ShapelyPrimitiveT = Union[
#    shapely.geometry.Point,
#    shapely.geometry.LineString,
#    shapely.geometry.Polygon
#]


#class ShapePrimitive(ShapeBase):
#    #@abstractmethod
#    #def _slice_primitive(self, start: Real, end: Real) -> "Shape":
#    #    pass

#    @lazy_property_initializer
#    @staticmethod
#    def _line_strings_() -> list[shapely.geometry.LineString]:
#        # Require at least one component
#        return NotImplemented

#    #@lazy_property
#    #@staticmethod
#    #def _line_strings_lengths_(line_strings: list[shapely.geometry.LineString]) -> FloatsT:
#    #    return np.array([max(line_string.length, 1e-6) for line_string in line_strings])

#    #@lazy_property
#    #@staticmethod
#    #def _line_strings_total_length_(line_strings_lengths: FloatsT) -> float:
#    #    return line_strings_lengths.sum()


#class PointPrimitive(ShapePrimitive):
#    def __init__(self, shape: shapely.geometry.Point):
#        super().__init__()
#        self._shapely_obj_ = shape

#    @lazy_property_initializer
#    @staticmethod
#    def _shapely_obj_() -> shapely.geometry.Point:
#        return NotImplemented

#    #def _slice_primitive(self, start: Real, end: Real) -> "Shape":
#    #    return Shape(self._shapely_obj_)

#    @lazy_property
#    @staticmethod
#    def _line_strings_(shapely_obj: shapely.geometry.Point) -> list[shapely.geometry.LineString]:
#        return [shapely.geometry.LineString([shapely_obj, shapely_obj])]


#class LineStringPrimitive(ShapePrimitive):
#    def __init__(self, shape: shapely.geometry.LineString):
#        super().__init__()
#        self._shapely_obj_ = shape

#    @lazy_property_initializer
#    @staticmethod
#    def _shapely_obj_() -> shapely.geometry.LineString:
#        return NotImplemented

#    #def _slice_primitive(self, start: Real, end: Real) -> "Shape":
#    #    return Shape(self._substring(self._shapely_obj_, start, end))

#    @lazy_property
#    @staticmethod
#    def _line_strings_(shapely_obj: shapely.geometry.LineString) -> list[shapely.geometry.LineString]:
#        return [shapely_obj]


#class PolygonPrimitive(ShapePrimitive):
#    def __init__(self, shape: shapely.geometry.Polygon):
#        super().__init__()
#        self._shapely_obj_ = shape

#    @lazy_property_initializer
#    @staticmethod
#    def _shapely_obj_() -> shapely.geometry.Polygon:
#        return NotImplemented

#    @lazy_property
#    @staticmethod
#    def _line_strings_(shapely_obj: shapely.geometry.Polygon) -> list[shapely.geometry.LineString]:
#        result: list[shapely.geometry.LineString] = []
#        if isinstance((exterior := shapely_obj.exterior), shapely.geometry.LinearRing):
#            result.append(exterior)
#        for interior in shapely_obj.interiors:
#            if isinstance(interior, shapely.geometry.LinearRing):
#                result.append(interior)
#        return result

#    #@lazy_property
#    #@staticmethod
#    #def _length_cumsum_list_(polygon_rings: list[shapely.geometry.LinearRing]) -> list[float]:
#    #    ring_lengths = np.array([ring.length for ring in polygon_rings])
#    #    total_length = ring_lengths.sum()
#    #    if not total_length:
#    #        return [0.0]
#    #    return [0.0, *(ring_lengths.cumsum() / total_length)]

#    #def _slice_primitive(self, start: Real, end: Real) -> "Shape":
#    #    polygon_rings = self._polygon_rings_
#    #    knots = self._length_cumsum_list_
#    #    start_index, start_residue = self._integer_interpolate(knots, start)
#    #    end_index, end_residue = self._integer_interpolate(knots, end)
#    #    if start_index == end_index:
#    #        components = [self._substring(polygon_rings[start_index], start_residue, end_residue)]
#    #    else:
#    #        components = [
#    #            self._substring(polygon_rings[start_index], start_residue, 1.0),
#    #            *polygon_rings[start_index + 1:end_index],
#    #            self._substring(polygon_rings[end_index], 1.0, end_residue)
#    #        ]
#    #    return Shape(reduce(shapely.geometry.base.BaseGeometry.__xor__, components))

#    #def _as_line_strings(self) -> list[shapely.geometry.LineString]:
#    #    return self._polygon_rings_


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
        return np.insert(lengths.cumsum() / length, 0, 0.0)

    @property
    @abstractmethod
    def _shapely_obj_(self) -> shapely.geometry.base.BaseGeometry:
        pass

    @abstractmethod
    def interpolate_point(self, alpha: Real) -> Vec2T:
        pass

    @abstractmethod
    def interpolate_shape(self, other: "ShapeInterpolantBase", alpha: Real) -> "ShapeInterpolantBase":
        pass

    @abstractmethod
    def partial(self, start: Real, end: Real) -> "ShapeInterpolantBase":
        pass


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

    #@lazy_property
    #@staticmethod
    #def _shapely_obj_(children: list[_ChildT]) -> shapely.geometry.base.BaseGeometry:
    #    return shapely.geometry.GeometryCollection([
    #        child._shapely_obj_ for child in children
    #    ])

    def interpolate_point(self, alpha: Real) -> Vec2T:
        index, residue = self._integer_interpolate(self._length_knots_, alpha)
        return self._children_[index].interpolate_point(residue)

    def interpolate_shape(self, other: "ShapeInterpolant[_ChildT]", alpha: Real) -> "ShapeInterpolant[_ChildT]":
        children = self._children_
        knots_0 = self._length_knots_[1:]
        knots_1 = other._length_knots_[1:]
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

        #all_knots = sorted(set(self._length_knots_) | set(other._length_knots_))
        #return MultiLineString([
        #    self.partial(start, end)._line_strings_[0].interpolate_shape(
        #        other.partial(start, end)._line_strings_[0], alpha  # wierd here
        #    )
        #    for start, end in it.pairwise(all_knots)
        #])

    def partial(self, start: Real, end: Real) -> "ShapeInterpolant[_ChildT]":
        children = self._children_
        knots = self._length_knots_
        start_index, start_residue = self._integer_interpolate(knots, start)
        end_index, end_residue = self._integer_interpolate(knots, end)
        if start_index == len(knots) - 1:
            start_index = len(knots) - 2
            start_residue = 1.0
        if end_index == len(knots) - 1:
            end_index = len(knots) - 2
            end_residue = 1.0
        if start_index == end_index:
            new_children = [children[start_index].partial(start_residue, end_residue)]
        else:
            new_children = [
                children[start_index].partial(start_residue, 1.0),
                *children[start_index + 1:end_index],
                children[end_index].partial(1.0, end_residue)
            ]
        return self.__class__(new_children)

    @classmethod
    def _integer_interpolate(cls, array: FloatsT, target: Real) -> tuple[int, float]:
        """
        Assumed that `array` is already sorted, and that `array[0] <= target <= array[-1]`
        Returns `(i, (target - array[i]) / (array[i + 1] - array[i]))` such that
        `0 <= i <= len(array) - 1` and `array[i] <= target < array[i + 1]`.
        """
        # TODO
        index = int(cls._interp1d(array, np.arange(len(array)).astype(float), kind="previous")(target))
        if index == len(array) - 1:
            return len(array) - 1, 0.0
        try:
            return index, (target - array[index]) / (array[index + 1] - array[index])
        except ZeroDivisionError:
            return index, 0.0

    @classmethod
    def _interp1d(cls, x: FloatsT, y: FloatsT, tol: Real = 1e-6, **kwargs) -> scipy.interpolate.interp1d:
        # Append one more sample point at each side in order to prevent from floating error.
        # Also solves the issue where we have only one sample, while the original function requires at least two.
        # Assumed that `x` is already sorted.
        new_x = np.array([x[0] - tol, *x, x[-1] + tol])
        new_y = np.array([y[0], *y, y[-1]])
        return scipy.interpolate.interp1d(new_x, new_y, **kwargs)


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
    def _shapely_line_string_(coords: Vec2sT) -> shapely.geometry.LineString:
        return shapely.geometry.LineString(coords)

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

    def interpolate_point(self, alpha: Real) -> Vec2T:
        return np.array(self._shapely_line_string_.interpolate_point(alpha, normalized=True).coords)

    def interpolate_shape(self, other: "LineString", alpha: Real) -> "LineString":
        all_knots = sorted(set(self._length_knots_) | set(other._length_knots_))
        return LineString(np.array([
            (1.0 - alpha) * self.interpolate_point(knot) + alpha * other.interpolate_point(knot)
            for knot in all_knots
        ]))

    def partial(self, start: Real, end: Real) -> "LineString":
        return LineString(np.array(
            shapely.ops.substring(self._shapely_line_string_, start, end, normalized=True).coords
        ))


#class MultiLineString(ShapeInterpolant[LineString]):
#    pass
    #def __init__(self, line_strings: list[LineString]):
    #    super().__init__()
    #    self._line_strings_ = line_strings

    #@lazy_property_initializer_writable
    #@staticmethod
    #def _line_strings_() -> list[LineString]:
    #    return NotImplemented

    #@lazy_property
    #@staticmethod
    #def _knot_item_lists_(length_knots: FloatsT, line_strings: list[LineString]) -> list[list[tuple[float, Vec2T]]]:
    #    return [
    #        [
    #            (knot_value * child_knot_value, coord)
    #            for child_knot_value, coord in zip(line_string._length_knots_, line_string._coords_)
    #        ]
    #        for knot_value, line_string in zip(length_knots[1:], line_strings)
    #    ]

    #@classmethod
    #def _from_knot_item_lists(cls, knot_item_lists: list[list[tuple[float, Vec2T]]]) -> "MultiLineString":
    #    return MultiLineString([
    #        LineString(np.array([
    #            coord for _, coord in knot_item_list
    #        ]))
    #        for knot_item_list in knot_item_lists
    #    ])

    #@classmethod
    #def _zip_multiple(cls, *multi_line_strings: "MultiLineString") -> list[list[tuple[float, tuple[Vec2T, ...]]]]:
    #    result: list[list[tuple[float, tuple[Vec2T, ...]]]] = []
    #    current: list[tuple[float, tuple[Vec2T, ...]]] = []
    #    all_knot_item_lists = [multi_line_string._knot_item_lists_ for multi_line_string in multi_line_strings]



class Polygon(ShapeInterpolant[LineString]):
    #def __init__(self, shell: LineString, holes: list[LineString] | None = None):
    #    if holes is None:
    #        holes = []
    #    super().__init__([shell, *holes])

    @lazy_property
    @staticmethod
    def _shapely_obj_(children: list[LineString]) -> shapely.geometry.Polygon:
        shell_coords = children[0]._coords_
        shell_coords_len = len(shell_coords)
        if not shell_coords_len:
            return shapely.geometry.base.EmptyGeometry()
        elif shell_coords_len == 1:
            return shapely.geometry.Point(shell_coords[0])
        elif shell_coords_len == 2:
            return shapely.geometry.LineString(shell_coords)
        return shapely.geometry.Polygon(
            shell_coords,
            [
                hole_coords
                for line_string in children[1:]
                if len(hole_coords := line_string._coords_) >= 3
            ]
        )

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
    def _shapely_obj_(children: list[Polygon]) -> shapely.geometry.base.BaseGeometry:
        return shapely.geometry.GeometryCollection([
            child._shapely_obj_ for child in children
        ])

    #def __init__(self, polygons: list[Polygon]):
    #    super().__init__()
    #    self._polygons_ = polygons

    #@lazy_property_initializer_writable
    #@staticmethod
    #def _polygons_() -> list[Polygon]:
    #    return NotImplemented

    #@lazy_property
    #@staticmethod
    #def _lengths_(polygons: list[Polygon]) -> FloatsT:
    #    return np.array([polygon._length_ for polygon in polygons])

        #polygons = self._polygons_
        #if not polygons:
        #    return Shape([])

        #knots = self._length_knots_
        #start_index, start_residue = self._integer_interpolate(knots, start)
        #end_index, end_residue = self._integer_interpolate(knots, end)
        #if start_index == end_index:
        #    components = [polygons[start_index].partial(start_residue, end_residue)]
        #else:
        #    components = [
        #        polygons[start_index].partial(start_residue, 1.0),
        #        *polygons[start_index + 1:end_index],
        #        polygons[end_index].partial(1.0, end_residue)
        #    ]
        #return Shape.from_shapely(reduce(shapely.geometry.base.BaseGeometry.__xor__, [
        #    polygon._shapely_obj_ for component in components for polygon in component._polygons_
        #]))

    #def interpolate(self, alpha: Real) -> Vec2T:
    #    index, residue = self._integer_interpolate(self._length_knots_, alpha)
    #    return self._line_strings_[index].interpolate(residue)

    #def interpolate_shape(self, other: "Shape", alpha: Real) -> "Shape":
    #    polygons = super().interpolate_shape(other, alpha)

        #all_knots = sorted(set(self._length_knots_) | set(other._length_knots_))
        #return MultiLineString([
        #    self.partial(start, end)._line_strings_[0].interpolate_shape(
        #        other.partial(start, end)._line_strings_[0], alpha  # wierd here
        #    )
        #    for start, end in it.pairwise(all_knots)
        #])

    #def _build_from_new_children(self, children: list[Polygon]) -> "Shape":
    #    return Shape(reduce(shapely.geometry.base.BaseGeometry.__xor__, [
    #        child._shapely_obj_ for child in children
    #    ]))

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
            polygon._shapely_obj_ for polygon in polygons._children_
        ]))
