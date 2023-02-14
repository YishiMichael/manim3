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
    Iterable,
    Literal,
    TypeVar
)

import numpy as np
from scipy.interpolate import BSpline
import shapely.geometry
import shapely.ops
import shapely.validation
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
    NewData,
    lazy_basedata,
    lazy_property
)
from ..utils.space import SpaceUtils


_VecT = TypeVar("_VecT", bound=Vec2T | Vec3T)
_VecsT = TypeVar("_VecsT", bound=Vec2sT | Vec3sT)


class ShapeInterpolant(Generic[_VecT, _VecsT], LazyBase):
    @lazy_basedata
    @staticmethod
    def _lengths_() -> FloatsT:
        # Make sure all entries are non-zero to avoid zero divisions
        return NotImplemented

    @lazy_property
    @staticmethod
    def _length_(lengths: FloatsT) -> float:
        return lengths.sum()

    @lazy_property
    @staticmethod
    def _length_knots_(lengths: FloatsT) -> FloatsT:
        if not len(lengths):
            return np.zeros(1)
        unnormalized_knots = np.insert(lengths.cumsum(), 0, 0.0)
        # Ensure the last entry is always precisely 1.0
        return unnormalized_knots / unnormalized_knots[-1]

    @abstractmethod
    def interpolate_point(self, alpha: Real) -> _VecT:
        pass

    def interpolate_points(self, alphas: Iterable[Real]) -> _VecsT:
        return np.array([self.interpolate_point(alpha) for alpha in alphas])

    @classmethod
    def _get_residue(cls, target: Real, array: FloatsT, index: int) -> float:
        try:
            return (target - array[index]) / (array[index + 1] - array[index])
        except ZeroDivisionError:
            return 0.0

    def _integer_interpolate(
        self,
        target: Real,
        *,
        side: Literal["left", "right"] = "right"
    ) -> tuple[int, float]:
        """
        Assumed that `array` is non-empty and already sorted, and that `0 = array[0] <= target <= array[-1]`
        Returns `(i, (target - array[i - 1]) / (array[i] - array[i - 1]))` such that
        `1 <= i <= len(array) - 1` and `array[i - 1] <= target <= array[i]`.
        """
        array = self._length_knots_
        assert len(array)
        index = int(np.searchsorted(array, target, side=side))
        if index == 0:
            return 1, 0.0
        if index == len(array):
            return len(array) - 1, 1.0
        return index, self._get_residue(target, array, index - 1)


class LineString(ShapeInterpolant[_VecT, _VecsT]):
    def __init__(self, coords: _VecsT):
        # TODO: shall we first remove redundant adjacent points?
        assert len(coords)
        super().__init__()
        self._coords_ = NewData(coords)

    @lazy_basedata
    @staticmethod
    def _coords_() -> _VecsT:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _kind_(coords: _VecsT) -> str:
        if len(coords) == 1:
            return "point"
        if np.isclose(SpaceUtils.norm(coords[-1] - coords[0]), 0.0):
            return "linear_ring"
        return "line_string"

    @lazy_property
    @staticmethod
    def _shapely_component_(kind: str, coords: _VecsT) -> shapely.geometry.base.BaseGeometry:
        if kind == "point":
            return shapely.geometry.Point(coords[0])
        if len(coords) == 2:
            return shapely.geometry.LineString(coords)
        return shapely.validation.make_valid(shapely.geometry.Polygon(coords))

    @lazy_property
    @staticmethod
    def _lengths_(coords: _VecsT) -> FloatsT:
        return np.maximum(SpaceUtils.norm(coords[1:] - coords[:-1]), 1e-6)

    @classmethod
    def _lerp(cls, vec_0: _VecT, vec_1: _VecT, alpha: Real) -> _VecT:
        return SpaceUtils.lerp(vec_0, vec_1, alpha)

    def interpolate_point(self, alpha: Real) -> _VecT:
        if self._kind_ == "point":
            return self._coords_[0]
        index, residue = self._integer_interpolate(alpha)
        return self._lerp(self._coords_[index - 1], self._coords_[index], residue)

    def interpolate_shape(self, other: "LineString", alpha: Real) -> "LineString":
        all_knots = np.unique(np.concatenate((self._length_knots_, other._length_knots_)))
        return LineString(np.array([
            self._lerp(self.interpolate_point(knot), other.interpolate_point(knot), alpha)
            for knot in all_knots
        ]))

    def partial(self, start: Real, stop: Real) -> "LineString":
        coords = self._coords_
        if self._kind_ == "point":
            new_coords = [coords[0]]
        else:
            start_index, start_residue = self._integer_interpolate(start)
            stop_index, stop_residue = self._integer_interpolate(stop)
            if start_index == stop_index and start_residue == stop_residue:
                new_coords = [
                    self._lerp(coords[start_index - 1], coords[start_index], start_residue)
                ]
            else:
                new_coords = [
                    self._lerp(coords[start_index - 1], coords[start_index], start_residue),
                    *coords[start_index:stop_index],
                    self._lerp(coords[stop_index - 1], coords[stop_index], stop_residue)
                ]
        return LineString(np.array(new_coords))


class MultiLineString(ShapeInterpolant[_VecT, _VecsT]):
    def __init__(self, children: list[LineString[_VecT, _VecsT]] | None = None):
        super().__init__()
        if children is not None:
            self._children_ = NewData(tuple(children))

    @lazy_basedata
    @staticmethod
    def _children_() -> tuple[LineString[_VecT, _VecsT], ...]:
        return ()

    @lazy_property
    @staticmethod
    def _lengths_(children: tuple[LineString[_VecT, _VecsT], ...]) -> FloatsT:
        return np.maximum(np.array([child._length_ for child in children]), 1e-6)

    def interpolate_point(self, alpha: Real) -> _VecT:
        index, residue = self._integer_interpolate(alpha)
        return self._children_[index - 1].interpolate_point(residue)

    def interpolate_shape(
        self,
        other: "MultiLineString[_VecT, _VecsT]",
        alpha: Real,
        *,
        has_mending: bool
    ):
        children_0 = self._children_
        children_1 = other._children_
        knots_0 = self._length_knots_
        knots_1 = other._length_knots_
        index_0 = 0
        index_1 = 0
        all_knots = np.unique(np.concatenate((knots_0, knots_1)))
        line_strings: list[LineString[_VecT, _VecsT]] = []
        for start_knot, stop_knot in it.pairwise(all_knots):

            start_residue_0 = self._get_residue(start_knot, knots_0, index_0)
            stop_residue_0 = self._get_residue(stop_knot, knots_0, index_0)
            child_0 = children_0[index_0].partial(start_residue_0, stop_residue_0)

            start_residue_1 = self._get_residue(start_knot, knots_1, index_1)
            stop_residue_1 = self._get_residue(stop_knot, knots_1, index_1)
            child_1 = children_1[index_1].partial(start_residue_1, stop_residue_1)

            line_strings.append(child_0.interpolate_shape(child_1, alpha))

            if knots_0[index_0 + 1] == stop_knot:
                index_0 += 1
            if knots_1[index_1 + 1] == stop_knot:
                index_1 += 1

        if has_mending:
            for index, (line_string, (start_knot, stop_knot)) in enumerate(
                zip(children_0, it.pairwise(knots_0), strict=True)
            ):
                coords = line_string.interpolate_points(
                    self._get_residue(knot, knots_0, index)
                    for knot in all_knots
                    if start_knot <= knot <= stop_knot
                )
                if len(coords) == 2:
                    continue
                coords_center = np.average(coords, axis=0)
                line_strings.append(LineString(SpaceUtils.lerp(coords, coords_center, alpha)))

            for index, (line_string, (start_knot, stop_knot)) in enumerate(
                zip(children_1, it.pairwise(knots_1), strict=True)
            ):
                coords = line_string.interpolate_points(
                    self._get_residue(knot, knots_1, index)
                    for knot in all_knots
                    if start_knot <= knot <= stop_knot
                )
                if len(coords) == 2:
                    continue
                coords_center = np.average(coords, axis=0)
                line_strings.append(LineString(SpaceUtils.lerp(coords_center, coords, alpha)))

        return self.__class__(line_strings)

    def partial(self, start: Real, stop: Real):
        children = self._children_
        if not children:
            return self.__class__()

        start_index, start_residue = self._integer_interpolate(start, side="right")
        stop_index, stop_residue = self._integer_interpolate(stop, side="left")
        if start_index == stop_index:
            new_children = [children[start_index - 1].partial(start_residue, stop_residue)]
        else:
            new_children = [
                children[start_index - 1].partial(start_residue, 1.0),
                *children[start_index:stop_index - 1],
                children[stop_index - 1].partial(0.0, stop_residue)
            ]
        return self.__class__(new_children)

    @classmethod
    def concatenate(cls, multi_line_strings: "Iterable[MultiLineString[_VecT, _VecsT]]"):
        return cls([
            line_string
            for multi_line_string in multi_line_strings
            for line_string in multi_line_string._children_
        ])


class LineString2D(LineString[Vec2T, Vec2sT]):
    pass


class LineString3D(LineString[Vec3T, Vec3sT]):
    pass


class MultiLineString2D(MultiLineString[Vec2T, Vec2sT]):
    pass


class MultiLineString3D(MultiLineString[Vec3T, Vec3sT]):
    pass


class Shape(LazyBase):
    def __init__(self, arg: MultiLineString2D | shapely.geometry.base.BaseGeometry | se.Shape | None = None):
        if arg is None:
            multi_line_string = None
        elif isinstance(arg, MultiLineString2D):
            multi_line_string = arg
        else:
            if isinstance(arg, shapely.geometry.base.BaseGeometry):
                coords_iter = self._iter_coords_from_shapely_obj(arg)
            elif isinstance(arg, se.Shape):
                coords_iter = self._iter_coords_from_se_shape(arg)
            else:
                raise TypeError(f"Cannot handle argument in Shape constructor: {arg}")
            multi_line_string = MultiLineString2D([
                LineString2D(coords)
                for coords in coords_iter
                if len(coords)
            ])

        super().__init__()
        if multi_line_string is not None:
            self._multi_line_string_ = NewData(multi_line_string)

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
        se_path = se.Path(se_shape.segments(transformed=True))
        se_path.approximate_arcs_with_cubics()
        coords_list: list[Vec2T] = []
        #current_contour_start_point: Vec2T = np.zeros(2)
        for segment in se_path.segments(transformed=True):
            if isinstance(segment, se.Move):
                yield np.array(coords_list)
                #current_contour_start_point = np.array(segment.end)
                coords_list = [np.array(segment.end)]
            elif isinstance(segment, se.Linear):  # Line & Close
                #coords_list.append(current_contour_start_point)
                #yield np.array(coords_list)
                #coords_list = []
                coords_list.append(np.array(segment.end))
                #elif isinstance(segment, se.Line):
                #    coords_list.append(np.array(segment.end))
            else:
                if isinstance(segment, se.QuadraticBezier):
                    control_points = [segment.start, segment.control, segment.end]
                elif isinstance(segment, se.CubicBezier):
                    control_points = [segment.start, segment.control1, segment.control2, segment.end]
                else:
                    raise ValueError(f"Cannot handle path segment type: {type(segment)}")
                coords_list.extend(cls._get_bezier_sample_points(np.array(control_points))[1:])
        yield np.array(coords_list)

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

    @classmethod
    def _to_shapely_object(cls, multi_line_string: MultiLineString2D) -> shapely.geometry.base.BaseGeometry:
        return reduce(shapely.geometry.base.BaseGeometry.__xor__, [
            line_string._shapely_component_
            for line_string in multi_line_string._children_
        ], shapely.geometry.GeometryCollection())

    def interpolate_point(self, alpha: Real) -> Vec2T:
        return self._multi_line_string_.interpolate_point(alpha)

    def interpolate_shape(
        self,
        other: "Shape",
        alpha: Real,
        *,
        has_mending: bool
    ) -> "Shape":
        multi_line_string = self._multi_line_string_.interpolate_shape(
            other._multi_line_string_,
            alpha,
            has_mending=has_mending
        )
        return Shape(Shape._to_shapely_object(multi_line_string))

    def partial(self, start: Real, stop: Real) -> "Shape":
        return Shape(self._multi_line_string_.partial(start, stop))

    @classmethod
    def concatenate(cls, shapes: "Iterable[Shape]") -> "Shape":
        return Shape(MultiLineString2D.concatenate(
            shape._multi_line_string_
            for shape in shapes
        ))

    #@classmethod
    #def interpolate_method(cls, shape_0: "Shape", shape_1: "Shape", alpha: Real) -> "Shape":
    #    return shape_0.interpolate_shape(shape_1, alpha)

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
