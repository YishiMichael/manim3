__all__ = [
    "LineString2D",
    "LineString3D",
    "LineStringKind",
    "MultiLineString2D",
    "MultiLineString3D",
    "Shape"
]


from abc import abstractmethod
from enum import Enum
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
    Vec2T,
    Vec3T,
    Vec2sT,
    Vec3sT
)
from ..lazy.core import (
    LazyCollection,
    LazyObject
)
from ..lazy.interface import (
    Lazy,
    LazyMode
)
from ..utils.space import SpaceUtils


_VecT = TypeVar("_VecT", bound=Vec2T | Vec3T)
_VecsT = TypeVar("_VecsT", bound=Vec2sT | Vec3sT)


class LineStringKind(Enum):
    POINT = 0
    LINE_STRING = 1
    LINEAR_RING = 2


class ShapeInterpolant(Generic[_VecT, _VecsT], LazyObject):
    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _lengths_(cls) -> FloatsT:
        # Make sure all entries are non-zero to avoid zero divisions
        return NotImplemented

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
    ) -> _VecT:
        pass

    def interpolate_points(
        self,
        alphas: Iterable[float]
    ) -> _VecsT:
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
        assert len(list_indices_groups) >= 2
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


class LineString(ShapeInterpolant[_VecT, _VecsT]):
    def __init__(
        self,
        coords: _VecsT
    ) -> None:
        # TODO: shall we first remove redundant adjacent points?
        assert len(coords)
        super().__init__()
        self._coords_ = coords

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _coords_(cls) -> _VecsT:
        return NotImplemented

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _kind_(
        cls,
        coords: _VecsT
    ) -> LineStringKind:
        if len(coords) == 1:
            return LineStringKind.POINT
        if np.isclose(SpaceUtils.norm(coords[-1] - coords[0]), 0.0):
            return LineStringKind.LINEAR_RING
        return LineStringKind.LINE_STRING

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _shapely_component_(
        cls,
        kind: str,
        coords: _VecsT
    ) -> shapely.geometry.base.BaseGeometry:
        if kind == LineStringKind.POINT:
            return shapely.geometry.Point(coords[0])
        if len(coords) == 2:
            return shapely.geometry.LineString(coords)
        return shapely.validation.make_valid(shapely.geometry.Polygon(coords))

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _lengths_(
        cls,
        coords: _VecsT
    ) -> FloatsT:
        return np.maximum(SpaceUtils.norm(coords[1:] - coords[:-1]), 1e-6)

    @classmethod
    def _lerp(
        cls,
        vec_0: _VecT,
        vec_1: _VecT,
        alpha: float
    ) -> _VecT:
        return SpaceUtils.lerp(vec_0, vec_1, alpha)

    def interpolate_point(
        self,
        alpha: float
    ) -> _VecT:
        coords = self._coords_.value
        if self._kind_.value == LineStringKind.POINT:
            return coords[0]
        index, residue = self._integer_interpolate(alpha)
        return self._lerp(coords[index - 1], coords[index], residue)

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
                    self._lerp(coords[start_index - 1], coords[start_index], start_residue)
                ]
            else:
                new_coords = [
                    self._lerp(coords[start_index - 1], coords[start_index], start_residue),
                    *coords[start_index:stop_index],
                    self._lerp(coords[stop_index - 1], coords[stop_index], stop_residue)
                ]
        return LineString(np.array(new_coords))

    def interpolate_shape_callback(
        self,
        other: "LineString"
    ) -> "Callable[[float], LineString]":
        all_knots = np.unique(np.concatenate((self._length_knots_.value, other._length_knots_.value)))
        point_callbacks: list[Callable[[float], _VecT]] = [
            SpaceUtils.lerp_callback(self.interpolate_point(knot), other.interpolate_point(knot))
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


class MultiLineString(ShapeInterpolant[_VecT, _VecsT]):
    def __init__(
        self,
        children: list[LineString[_VecT, _VecsT]] | None = None
    ) -> None:
        super().__init__()
        if children is not None:
            self._children_.add(*children)

    @Lazy.variable(LazyMode.COLLECTION)
    @classmethod
    def _children_(cls) -> LazyCollection[LineString[_VecT, _VecsT]]:
        return LazyCollection()

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _lengths_(
        cls,
        children__length: list[float]
    ) -> FloatsT:
        return np.maximum(np.array(children__length), 1e-6)

    def interpolate_point(
        self,
        alpha: float
    ) -> _VecT:
        if not self._children_:
            raise ValueError("Attempting to interpolate an empty MultiLineString")
        index, residue = self._integer_interpolate(alpha)
        return self._children_[index - 1].interpolate_point(residue)

    def partial(
        self,
        start: float,
        stop: float
    ):
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

    def interpolate_shape_callback(
        self,
        other: "MultiLineString[_VecT, _VecsT]",
        *,
        has_inlay: bool
    ) -> "Callable[[float], MultiLineString[_VecT, _VecsT]]":
        children_0 = self._children_
        children_1 = other._children_
        if not children_0 or not children_1:
            raise ValueError("Attempting to interpolate an empty MultiLineString")

        (residue_list_list_0, residue_list_list_1), triplet_tuple_list = self._zip_knots(
            self._length_knots_.value, other._length_knots_.value
        )
        line_string_callbacks: list[Callable[[float], LineString[_VecT, _VecsT]]] = [
            children_0[index_0].partial(start_residue_0, stop_residue_0).interpolate_shape_callback(
                children_1[index_1].partial(start_residue_1, stop_residue_1)
            )
            for (index_0, start_residue_0, stop_residue_0), (index_1, start_residue_1, stop_residue_1) in triplet_tuple_list
        ]

        inlay_callbacks: list[Callable[[float], _VecsT]] = []
        for index_0, residues in enumerate(residue_list_list_0):
            coords = children_0[index_0].interpolate_points(residues)
            if len(coords) == 2:
                continue
            coords_center = np.average(coords, axis=0)
            inlay_callbacks.append(SpaceUtils.lerp_callback(coords, coords_center))

        for index_1, residues in enumerate(residue_list_list_1):
            coords = children_1[index_1].interpolate_points(residues)
            if len(coords) == 2:
                continue
            coords_center = np.average(coords, axis=0)
            inlay_callbacks.append(SpaceUtils.lerp_callback(coords_center, coords))

        def callback(
            alpha: float
        ) -> MultiLineString[_VecT, _VecsT]:
            line_strings = [
                line_string_callback(alpha)
                for line_string_callback in line_string_callbacks
            ]
            if has_inlay:
                line_strings.extend(
                    LineString(inlay_callback(alpha))
                    for inlay_callback in inlay_callbacks
                )
            return self.__class__(line_strings)
        return callback

    @classmethod
    def concatenate(
        cls,
        multi_line_strings: "Iterable[MultiLineString[_VecT, _VecsT]]"
    ):
        return cls(list(it.chain(*(
            multi_line_string._children_
            for multi_line_string in multi_line_strings
        ))))


class LineString2D(LineString[Vec2T, Vec2sT]):
    pass


class LineString3D(LineString[Vec3T, Vec3sT]):
    pass


class MultiLineString2D(MultiLineString[Vec2T, Vec2sT]):
    pass


class MultiLineString3D(MultiLineString[Vec3T, Vec3sT]):
    pass


class Shape(LazyObject):
    def __init__(
        self,
        arg: MultiLineString2D | shapely.geometry.base.BaseGeometry | se.Shape | None = None
    ) -> None:
        super().__init__()

        if arg is None:
            return
        if isinstance(arg, MultiLineString2D):
            multi_line_string = arg
        else:
            if isinstance(arg, shapely.geometry.base.BaseGeometry):
                coords_iter = self._iter_coords_from_shapely_obj(arg)
                self._precalculated_shapely_obj_ = arg
            elif isinstance(arg, se.Shape):
                coords_iter = self._iter_coords_from_se_shape(arg)
            else:
                raise TypeError(f"Cannot handle argument in Shape constructor: {arg}")
            multi_line_string = MultiLineString2D([
                LineString2D(coords)
                for coords in coords_iter
                if len(coords)
            ])
        self._multi_line_string_ = multi_line_string

    def __and__(
        self,
        other: "Shape"
    ):
        return self.intersection(other)

    def __or__(
        self,
        other: "Shape"
    ):
        return self.union(other)

    def __sub__(
        self,
        other: "Shape"
    ):
        return self.difference(other)

    def __xor__(
        self,
        other: "Shape"
    ):
        return self.symmetric_difference(other)

    @Lazy.variable(LazyMode.OBJECT)
    @classmethod
    def _multi_line_string_(cls) -> MultiLineString2D:
        return MultiLineString2D()

    @Lazy.variable(LazyMode.UNWRAPPED)
    @classmethod
    def _precalculated_shapely_obj_(cls) -> shapely.geometry.base.BaseGeometry | None:
        return None

    @Lazy.property(LazyMode.OBJECT)
    @classmethod
    def _multi_line_string_3d_(
        cls,
        _multi_line_string_: MultiLineString2D
    ) -> MultiLineString3D:
        return MultiLineString3D([
            LineString3D(SpaceUtils.increase_dimension(line_string._coords_.value))
            for line_string in _multi_line_string_._children_
        ])

    @Lazy.property(LazyMode.UNWRAPPED)
    @classmethod
    def _shapely_obj_(
        cls,
        _multi_line_string_: MultiLineString2D,
        precalculated_shapely_obj: shapely.geometry.base.BaseGeometry | None
    ) -> shapely.geometry.base.BaseGeometry:
        if precalculated_shapely_obj is not None:
            return precalculated_shapely_obj
        return cls._to_shapely_object(_multi_line_string_)

    @classmethod
    def _iter_coords_from_shapely_obj(
        cls,
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
            for child in shapely_obj.geoms:
                yield from cls._iter_coords_from_shapely_obj(child)
        else:
            raise TypeError

    @classmethod
    def _iter_coords_from_se_shape(
        cls,
        se_shape: se.Shape
    ) -> Generator[Vec2sT, None, None]:
        se_path = se.Path(se_shape.segments(transformed=True))
        se_path.approximate_arcs_with_cubics()
        coords_list: list[Vec2T] = []
        for segment in se_path.segments(transformed=True):
            if isinstance(segment, se.Move):
                yield np.array(coords_list)
                coords_list = [np.array(segment.end)]
            elif isinstance(segment, se.Linear):  # Line & Close
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

    @classmethod
    def _get_bezier_sample_points(
        cls,
        control_points: Vec2sT
    ) -> Vec2sT:
        def smoothen_samples(
            curve: Callable[[FloatsT], Vec2sT],
            samples: FloatsT,
            bisect_depth: int
        ) -> FloatsT:
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

    @classmethod
    def _to_shapely_object(
        cls,
        multi_line_string: MultiLineString2D
    ) -> shapely.geometry.base.BaseGeometry:
        return reduce(shapely.geometry.base.BaseGeometry.__xor__, [
            line_string._shapely_component_.value
            for line_string in multi_line_string._children_
        ], shapely.geometry.GeometryCollection())

    def interpolate_point(
        self,
        alpha: float
    ) -> Vec2T:
        return self._multi_line_string_.interpolate_point(alpha)

    def partial(
        self,
        start: float,
        stop: float
    ):
        cls = self.__class__
        return cls(self._multi_line_string_.partial(start, stop))

    def interpolate_shape_callback(
        self,
        other: "Shape",
        *,
        has_inlay: bool
    ) -> "Callable[[float], Shape]":
        cls = self.__class__
        multi_line_string_callback = self._multi_line_string_.interpolate_shape_callback(
            other._multi_line_string_,
            has_inlay=has_inlay
        )

        def callback(
            alpha: float
        ):
            multi_line_string = multi_line_string_callback(alpha)
            return cls(cls._to_shapely_object(multi_line_string))
        return callback

    @classmethod
    def concatenate(
        cls,
        shapes: "Iterable[Shape]"
    ):
        return cls(MultiLineString2D.concatenate(
            shape._multi_line_string_
            for shape in shapes
        ))

    # operations ported from shapely

    @property
    def area(self) -> float:
        return self._shapely_obj_.value.area

    def distance(
        self,
        other: "Shape"
    ) -> float:
        return self._shapely_obj_.value.distance(other._shapely_obj_.value)

    def hausdorff_distance(
        self,
        other: "Shape"
    ) -> float:
        return self._shapely_obj_.value.hausdorff_distance(other._shapely_obj_.value)

    @property
    def length(self) -> float:
        return self._shapely_obj_.value.length

    @property
    def centroid(self) -> Vec2T:
        return np.array(self._shapely_obj_.value.centroid)

    @property
    def convex_hull(self):
        cls = self.__class__
        return cls(self._shapely_obj_.value.convex_hull)

    @property
    def envelope(self):
        cls = self.__class__
        return cls(self._shapely_obj_.value.envelope)

    def buffer(
        self,
        distance: float,
        quad_segs: int = 16,
        cap_style: str = "round",
        join_style: str = "round",
        mitre_limit: float = 5.0,
        single_sided: bool = False
    ):
        cls = self.__class__
        return cls(self._shapely_obj_.value.buffer(
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
    ):
        cls = self.__class__
        return cls(self._shapely_obj_.value.intersection(other._shapely_obj_.value))

    def union(
        self,
        other: "Shape"
    ):
        cls = self.__class__
        return cls(self._shapely_obj_.value.union(other._shapely_obj_.value))

    def difference(
        self,
        other: "Shape"
    ):
        cls = self.__class__
        return cls(self._shapely_obj_.value.difference(other._shapely_obj_.value))

    def symmetric_difference(
        self,
        other: "Shape"
    ):
        cls = self.__class__
        return cls(self._shapely_obj_.value.symmetric_difference(other._shapely_obj_.value))
