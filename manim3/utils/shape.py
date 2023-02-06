__all__ = [
    "LineString2D",
    "LineString3D",
    "MultiLineString2D",
    "MultiLineString3D",
    "Shape"
]


from abc import abstractmethod
from functools import reduce
import itertools as it
from typing import (
    Callable,
    Generator,
    Generic,
    TypeVar
)

import numpy as np
from scipy.interpolate import BSpline
import shapely.geometry
import shapely.ops
import svgelements as se

from ..custom_typing import (
    FloatsT,
    Real,
    Vec2T,
    Vec3T,
    Vec2sT,
    Vec3sT
)
from ..utils.lazy import (
    LazyBase,
    LazyData,
    lazy_basedata,
    lazy_property
)
from ..utils.space import SpaceUtils


_VecT = TypeVar("_VecT", bound=Vec2T | Vec3T)
_VecsT = TypeVar("_VecsT", bound=Vec2sT | Vec3sT)


class ShapeInterpolant(Generic[_VecT, _VecsT], LazyBase):
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
    def interpolate_point(self, alpha: Real) -> _VecT:
        pass

    @abstractmethod
    def interpolate_shape(self, other: "ShapeInterpolant", alpha: Real) -> "ShapeInterpolant":
        pass

    @abstractmethod
    def partial(self, start: Real, end: Real) -> "ShapeInterpolant":
        pass

    @classmethod
    def _integer_interpolate(cls, array: FloatsT, target: Real) -> tuple[int, float]:
        """
        Assumed that `array` is already sorted, and that `0 <= array[0] <= target <= array[-1]`
        Returns `(i, (target - array[i - 1]) / (array[i] - array[i - 1]))` such that
        `0 <= i <= len(array) - 1` and `array[i - 1] <= target <= array[i]`,
        where we've interpreted `array[-1]` as 0.
        """
        if not len(array):
            return 0, 0.0
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


class LineString(ShapeInterpolant[_VecT, _VecsT]):
    def __new__(cls, coords: _VecsT):
        # TODO: shall we first remove redundant adjacent points?
        assert len(coords)
        #if coords.shape[1] == 3:
        #    raise ValueError
        #if cls.__name__ == "LineString2D":
        #    print(coords)
        instance = super().__new__(cls)
        instance._coords_ = LazyData(coords)
        return instance

    @lazy_basedata
    @staticmethod
    def _coords_() -> _VecsT:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _kind_(coords: _VecsT) -> str:
        if len(coords) == 1:
            return "point"
        if not np.isclose(SpaceUtils.norm(coords[-1] - coords[0]), 0.0):
            return "line_string"
        return "linear_ring"

    @lazy_property
    @staticmethod
    def _shapely_component_(kind: str, coords: _VecsT) -> shapely.geometry.base.BaseGeometry:
        if kind == "point":
            return shapely.geometry.Point(coords[0])
        if len(coords) == 2:
            return shapely.geometry.LineString(coords)
        return shapely.geometry.Polygon(coords)

    @lazy_property
    @staticmethod
    def _lengths_(coords: _VecsT) -> FloatsT:
        return SpaceUtils.norm(coords[1:] - coords[:-1])

    @classmethod
    def _lerp(cls, vec_0: _VecT, vec_1: _VecT, alpha: Real) -> _VecT:
        return SpaceUtils.lerp(vec_0, vec_1, alpha)

    def interpolate_point(self, alpha: Real) -> _VecT:
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
        if start_index == end_index and start_residue == end_residue:
            new_coords = [
                self._lerp(coords[start_index], coords[start_index + 1], start_residue)
            ]
        else:
            new_coords = [
                self._lerp(coords[start_index], coords[start_index + 1], start_residue),
                *coords[start_index + 1:end_index + 1],
                self._lerp(coords[end_index], coords[end_index + 1], end_residue)
            ]
        return LineString(np.array(new_coords))


class MultiLineString(ShapeInterpolant[_VecT, _VecsT]):
    def __new__(cls, children: list[LineString[_VecT, _VecsT]] | None = None):
        instance = super().__new__(cls)
        if children is not None:
            children_list = list(instance._children_)
            children_list.extend(children)
            instance._children_ = LazyData(tuple(children_list))
        return instance

    @lazy_basedata
    @staticmethod
    def _children_() -> tuple[LineString[_VecT, _VecsT], ...]:
        return ()

    @lazy_property
    @staticmethod
    def _lengths_(children: tuple[LineString[_VecT, _VecsT], ...]) -> FloatsT:
        return np.array([child._length_ for child in children])

    def interpolate_point(self, alpha: Real) -> _VecT:
        index, residue = self._integer_interpolate(self._length_knots_, alpha)
        return self._children_[index].interpolate_point(residue)

    def interpolate_shape(self, other: "MultiLineString[_VecT, _VecsT]", alpha: Real):
        children = self._children_
        knots_0 = self._length_knots_
        knots_1 = other._length_knots_
        current_knot = 0.0
        start_0 = 0.0
        start_1 = 0.0
        ptr_0 = 0
        ptr_1 = 0
        new_children: list[LineString[_VecT, _VecsT]] = []
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
        if not children:
            return self.__class__()

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


class LineString2D(LineString[Vec2T, Vec2sT]):
    pass


class LineString3D(LineString[Vec3T, Vec3sT]):
    pass


class MultiLineString2D(MultiLineString[Vec2T, Vec2sT]):
    pass


class MultiLineString3D(MultiLineString[Vec3T, Vec3sT]):
    pass


class Shape(LazyBase):
    def __new__(cls, arg: MultiLineString2D | shapely.geometry.base.BaseGeometry | se.Shape | None = None):
        instance = super().__new__(cls)
        if arg is None:
            multi_line_string = None
        elif isinstance(arg, MultiLineString2D):
            multi_line_string = arg
        else:
            if isinstance(arg, shapely.geometry.base.BaseGeometry):
                coords_iter = cls._iter_coords_from_shapely_obj(arg)
            elif isinstance(arg, se.Shape):
                coords_iter = cls._iter_coords_from_se_shape(arg)
            else:
                raise TypeError(f"Cannot handle argument in Shape constructor: {arg}")
            #print(list(coords_iter))
            multi_line_string = MultiLineString2D([
                LineString2D(coords)
                for coords in coords_iter
                if len(coords)
            ])
            #print(">>>")
            #for p in cls._iter_coords_from_se_shape(arg):
            #    print(p)
            #print("<<<")
            #for child in multi_line_string._children_:
            #    print(type(child), child._coords_)
            #print()
        if multi_line_string is not None:
            instance._multi_line_string_ = LazyData(multi_line_string)
        return instance

    def __and__(self, other: "Shape"):
        return self.intersection(other)

    def __or__(self, other: "Shape"):
        return self.union(other)

    def __sub__(self, other: "Shape"):
        return self.difference(other)

    def __xor__(self, other: "Shape"):
        return self.symmetric_difference(other)

    @lazy_basedata
    @staticmethod
    def _multi_line_string_() -> MultiLineString2D:
        return MultiLineString2D()

    @lazy_property
    @staticmethod
    def _multi_line_string_3d_(multi_line_string: MultiLineString2D) -> MultiLineString3D:
        return MultiLineString3D([
            LineString3D(SpaceUtils.increase_dimension(line_string._coords_))
            for line_string in multi_line_string._children_
        ])

    #@classmethod
    #def _get_multi_line_string_from_shapely_obj(cls, shapely_obj: shapely.geometry.base.BaseGeometry) -> MultiLineString2D:
    #    return MultiLineString2D(list(cls._iter_line_strings_from_shapely_obj(shapely_obj)))

    @classmethod
    def _iter_coords_from_shapely_obj(cls, shapely_obj: shapely.geometry.base.BaseGeometry) -> Generator[Vec2sT, None, None]:
        if isinstance(shapely_obj, shapely.geometry.Point | shapely.geometry.LineString):
            yield np.array(shapely_obj.coords)
        elif isinstance(shapely_obj, shapely.geometry.Polygon):
            shapely_obj = shapely.geometry.polygon.orient(shapely_obj)  # TODO: needed?
            yield np.array(shapely_obj.exterior.coords)
            for interior in shapely_obj.interiors:
                yield np.array(interior.coords)
        elif isinstance(shapely_obj, shapely.geometry.base.BaseMultipartGeometry):
            for child in shapely_obj.geoms:
                yield from cls._iter_coords_from_shapely_obj(child)
        else:
            raise TypeError

    @classmethod
    def _iter_coords_from_se_shape(cls, se_shape: se.Shape) -> Generator[Vec2sT, None, None]:
        #if isinstance(se_shape, str):
        #    se_shape = se.Path(se_shape)
        se_path = se.Path(se_shape.segments(transformed=True))
        se_path.approximate_arcs_with_cubics()
        #point_lists: list[list[Vec2T]] = []
        coords_list: list[Vec2T] = []
        current_path_start_point: Vec2T = np.zeros(2)
        for segment in se_path.segments(transformed=True):
            if isinstance(segment, se.Move):
                yield np.array(coords_list)
                #point_lists.append(coords_list)
                current_path_start_point = np.array(segment.end)
                coords_list = [current_path_start_point]
            elif isinstance(segment, se.Close):
                coords_list.append(current_path_start_point)
                yield np.array(coords_list)
                #point_lists.append(coords_list)
                coords_list = []
            elif isinstance(segment, se.Line):
                coords_list.append(np.array(segment.end))
            else:
                if isinstance(segment, se.QuadraticBezier):
                    control_points = [segment.start, segment.control, segment.end]
                elif isinstance(segment, se.CubicBezier):
                    control_points = [segment.start, segment.control1, segment.control2, segment.end]
                else:
                    raise ValueError(f"Cannot handle path segment type: {type(segment)}")
                coords_list.extend(cls._get_bezier_sample_points(np.array(control_points))[1:])
        yield np.array(coords_list)
        #point_lists.append(coords)

        #return MultiLineString2D([
        #    LineString2D(np.array(coords))
        #    for coords in point_lists if coords
        #])

    @classmethod
    def _get_bezier_sample_points(cls, control_points: Vec2sT) -> Vec2sT:
        def smoothen_samples(curve: Callable[[FloatsT], Vec2sT], samples: FloatsT, bisect_depth: int) -> FloatsT:
            # Bisect a segment if one of its endpoints has a turning angle above the threshold.
            # Bisect for no more than 4 times, so each curve will be split into no more than 16 segments.
            if bisect_depth == 4:
                return samples
            points = curve(samples)
            directions = SpaceUtils.normalize(points[1:] - points[:-1])
            angles = abs(np.arccos((directions[1:] * directions[:-1]).sum(axis=1)))
            large_angle_indices = np.squeeze(np.argwhere(angles > np.pi / 16.0), axis=1)
            if not len(large_angle_indices):
                return samples
            insertion_index_pairs = np.array(list(dict.fromkeys(it.chain(*(
                ((i, i + 1), (i + 1, i + 2))
                for i in large_angle_indices
            )))))
            new_samples = np.average(samples[insertion_index_pairs], axis=1)
            return smoothen_samples(curve, np.sort(np.concatenate((samples, new_samples))), bisect_depth + 1)

        order = len(control_points) - 1
        gamma = BSpline(
            t=np.append(np.zeros(order + 1), np.ones(order + 1)),
            c=control_points,
            k=order
        )
        if np.isclose(SpaceUtils.norm(gamma(1.0) - gamma(0.0)), 0.0):
            return np.array((gamma(0.0),))
        samples = smoothen_samples(gamma, np.linspace(0.0, 1.0, 3), 1)
        return gamma(samples).astype(float)

    @lazy_property
    @staticmethod
    def _shapely_obj_(multi_line_string: MultiLineString2D) -> shapely.geometry.base.BaseGeometry:
        return Shape._to_shapely_object(multi_line_string)

    #@lazy_property
    #@staticmethod
    #def _shapely_boundary_(polygons: Polygons) -> list[shapely.geometry.LineString]:
    #    return polygons._shapely_boundary_

    @classmethod
    def _to_shapely_object(cls, multi_line_string: MultiLineString2D) -> shapely.geometry.base.BaseGeometry:
        return reduce(shapely.geometry.base.BaseGeometry.__xor__, [
            line_string._shapely_component_
            for line_string in multi_line_string._children_
        ], shapely.geometry.GeometryCollection())

    def interpolate_point(self, alpha: Real) -> Vec2T:
        return self._multi_line_string_.interpolate_point(alpha)

    def interpolate_shape(self, other: "Shape", alpha: Real) -> "Shape":
        return Shape(self._multi_line_string_.interpolate_shape(other._multi_line_string_, alpha))

    def partial(self, start: Real, end: Real) -> "Shape":
        return Shape(self._multi_line_string_.partial(start, end))

    @classmethod
    def interpolate_method(cls, shape_0: "Shape", shape_1: "Shape", alpha: Real) -> "Shape":
        return shape_0.interpolate_shape(shape_1, alpha)

    #@lazy_property
    #@staticmethod
    #def _triangulation_(polygons: Polygons) -> tuple[VertexIndexType, Vec2sT]:
    #    return polygons._triangulation_

    # operations ported from shapely

    @property
    def area(self) -> float:
        return self._shapely_obj_.area

    def distance(self, other: "Shape") -> float:
        return self._shapely_obj_.distance(other._shapely_obj_)

    def hausdorff_distance(self, other: "Shape") -> float:
        return self._shapely_obj_.hausdorff_distance(other._shapely_obj_)

    @property
    def length(self) -> float:
        return self._shapely_obj_.length

    @property
    def centroid(self) -> Vec2T:
        return np.array(self._shapely_obj_.centroid)

    @property
    def convex_hull(self) -> "Shape":
        return Shape(self._shapely_obj_.convex_hull)

    @property
    def envelope(self) -> "Shape":
        return Shape(self._shapely_obj_.envelope)

    def buffer(
        self,
        distance: Real,
        quad_segs: int = 16,
        cap_style: str = "round",
        join_style: str = "round",
        mitre_limit: Real = 5.0,
        single_sided: bool = False
    ) -> "Shape":
        return Shape(self._shapely_obj_.buffer(
            distance=distance,
            quad_segs=quad_segs,
            cap_style=cap_style,
            join_style=join_style,
            mitre_limit=mitre_limit,
            single_sided=single_sided
        ))

    def intersection(self, other: "Shape") -> "Shape":
        return Shape(self._shapely_obj_.intersection(other._shapely_obj_))

    def union(self, other: "Shape") -> "Shape":
        return Shape(self._shapely_obj_.union(other._shapely_obj_))

    def difference(self, other: "Shape") -> "Shape":
        return Shape(self._shapely_obj_.difference(other._shapely_obj_))

    def symmetric_difference(self, other: "Shape") -> "Shape":
        return Shape(self._shapely_obj_.symmetric_difference(other._shapely_obj_))
