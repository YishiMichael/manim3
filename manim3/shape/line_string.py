from abc import abstractmethod
import itertools as it
from typing import (
    Callable,
    Iterable,
    Literal
)

import numpy as np

from ..custom_typing import (
    FloatsT,
    Vec3T,
    Vec3sT
)
from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..utils.space import SpaceUtils


class ShapeInterpolant(LazyObject):
    __slots__ = ()

    @Lazy.property_external
    @classmethod
    def _lengths_(cls) -> FloatsT:
        # Make sure all entries are non-zero to avoid zero divisions
        return np.zeros((0, 1))

    @Lazy.property_external
    @classmethod
    def _length_(
        cls,
        lengths: FloatsT
    ) -> float:
        return lengths.sum()

    @Lazy.property_external
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
        points: Vec3sT,
        *,
        is_ring: bool
    ) -> None:
        assert len(points)
        super().__init__()
        self._points_ = points
        self._is_ring_ = is_ring

    @Lazy.variable_external
    @classmethod
    def _points_(cls) -> Vec3sT:
        return np.zeros((0, 3))

    @Lazy.variable_shared
    @classmethod
    def _is_ring_(cls) -> bool:
        return False

    @Lazy.property_external
    @classmethod
    def _points_len_(
        cls,
        points: Vec3sT
    ) -> int:
        return len(points)

    @Lazy.property_external
    @classmethod
    def _path_points_(
        cls,
        points: Vec3sT,
        is_ring: bool
    ) -> Vec3sT:
        if not is_ring:
            return points.copy()
        return np.append(points, (points[0],), axis=0)

    @Lazy.property_external
    @classmethod
    def _lengths_(
        cls,
        path_points: Vec3sT
    ) -> FloatsT:
        return np.maximum(SpaceUtils.norm(path_points[1:] - path_points[:-1]), 1e-6)

    def interpolate_point(
        self,
        alpha: float
    ) -> Vec3T:
        path_points = self._path_points_.value
        if len(path_points) == 1:
            return path_points[0]
        index, residue = self._integer_interpolate(alpha)
        return SpaceUtils.lerp(path_points[index - 1], path_points[index])(residue)

    def partial(
        self,
        start: float,
        stop: float
    ) -> "LineString":
        assert start <= stop
        path_points = self._path_points_.value
        if len(path_points) == 1:
            new_points = [path_points[0]]
        else:
            start_index, start_residue = self._integer_interpolate(start, side="right")
            stop_index, stop_residue = self._integer_interpolate(stop, side="left")
            if start_index == stop_index and start_residue == stop_residue:
                new_points = [
                    SpaceUtils.lerp(path_points[start_index - 1], path_points[start_index])(start_residue)
                ]
            else:
                new_points = [
                    SpaceUtils.lerp(path_points[start_index - 1], path_points[start_index])(start_residue),
                    *path_points[start_index:stop_index],
                    SpaceUtils.lerp(path_points[stop_index - 1], path_points[stop_index])(stop_residue)
                ]
        return LineString(np.array(new_points), is_ring=False)

    @classmethod
    def get_interpolator(
        cls,
        line_string_0: "LineString",
        line_string_1: "LineString"
    ) -> "Callable[[float], LineString]":
        all_knots = np.unique(np.concatenate((line_string_0._length_knots_.value, line_string_1._length_knots_.value)))
        is_ring = False
        if line_string_0._is_ring_.value and line_string_1._is_ring_.value:
            all_knots = all_knots[:-1]
            is_ring = True

        point_interpolators = [
            SpaceUtils.lerp(line_string_0.interpolate_point(knot), line_string_1.interpolate_point(knot))
            for knot in all_knots
        ]

        def interpolator(
            alpha: float
        ) -> LineString:
            return LineString(np.array([
                point_interpolator(alpha)
                for point_interpolator in point_interpolators
            ]), is_ring=is_ring)

        return interpolator


class MultiLineString(ShapeInterpolant):
    __slots__ = ()

    def __init__(
        self,
        line_strings: Iterable[LineString] | None = None
    ) -> None:
        super().__init__()
        if line_strings is not None:
            self._line_strings_.extend(line_strings)

    @Lazy.variable_collection
    @classmethod
    def _line_strings_(cls) -> list[LineString]:
        return []

    @Lazy.property_external
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
        assert start <= stop
        result = MultiLineString()
        line_strings = self._line_strings_
        if not line_strings:
            return result

        start_index, start_residue = self._integer_interpolate(start, side="right")
        stop_index, stop_residue = self._integer_interpolate(stop, side="left")
        if start_index == stop_index:
            result._line_strings_.append(line_strings[start_index - 1].partial(start_residue, stop_residue))
        else:
            result._line_strings_.extend((
                line_strings[start_index - 1].partial(start_residue, 1.0),
                *line_strings[start_index:stop_index - 1],
                line_strings[stop_index - 1].partial(0.0, stop_residue)
            ))
        return result

    def get_partialor(self) -> "Callable[[float, float], MultiLineString]":
        return self.partial

    @classmethod
    def get_interpolator(
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

        (residue_list_list_0, residue_list_list_1), triplet_tuple_list = cls._zip_knots(
            multi_line_string_0._length_knots_.value, multi_line_string_1._length_knots_.value
        )
        line_string_interpolators: list[Callable[[float], LineString]] = [
            LineString.get_interpolator(
                line_strings_0[index_0].partial(start_residue_0, stop_residue_0),
                line_strings_1[index_1].partial(start_residue_1, stop_residue_1)
            )
            for (index_0, start_residue_0, stop_residue_0), (index_1, start_residue_1, stop_residue_1) in triplet_tuple_list
        ]

        inlay_interpolators: list[Callable[[float], Vec3sT]] = []
        for index_0, residues in enumerate(residue_list_list_0):
            points = line_strings_0[index_0].interpolate_points(residues)
            if len(points) == 2:
                continue
            points_center: Vec3T = np.average(points, axis=0)
            inlay_interpolators.append(SpaceUtils.lerp(points, points_center))

        for index_1, residues in enumerate(residue_list_list_1):
            points = line_strings_1[index_1].interpolate_points(residues)
            if len(points) == 2:
                continue
            points_center: Vec3T = np.average(points, axis=0)
            inlay_interpolators.append(SpaceUtils.lerp(points_center, points))

        def interpolator(
            alpha: float
        ) -> MultiLineString:
            result = MultiLineString()
            result._line_strings_.extend(
                line_string_interpolator(alpha)
                for line_string_interpolator in line_string_interpolators
            )
            if has_inlay:
                result._line_strings_.extend(
                    LineString(inlay_interpolator(alpha), is_ring=True)
                    for inlay_interpolator in inlay_interpolators
                )
            return result
        return interpolator

    @classmethod
    def concatenate(
        cls,
        multi_line_strings: "Iterable[MultiLineString]"
    ) -> "MultiLineString":
        result = MultiLineString()
        result._line_strings_.extend(it.chain.from_iterable(
            multi_line_string._line_strings_
            for multi_line_string in multi_line_strings
        ))
        return result
