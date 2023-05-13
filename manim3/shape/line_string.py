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

    @classmethod
    def _get_residue(
        cls,
        target: float,
        array: FloatsT,
        index: int
    ) -> float:
        return (target - array[index]) / (array[index + 1] - array[index])

    @classmethod
    def _integer_interpolate(
        cls,
        target: float,
        array: FloatsT,
        *,
        side: Literal["left", "right"] = "right"
    ) -> tuple[int, float]:
        """
        Assumed that `array` has at least 2 elements and already sorted, and that `0 = array[0] <= target <= array[-1]`.
        Returns `(i, (target - array[i - 1]) / (array[i] - array[i - 1]))` such that
        `1 <= i <= len(array) - 1` and `array[i - 1] <= target <= array[i]`.
        """
        #array = self._length_knots_.value
        assert len(array) >= 2
        index = int(np.searchsorted(array, target, side=side))
        if index == 0:
            return 1, 0.0
        if index == len(array):
            return len(array) - 1, 1.0
        return index, cls._get_residue(target, array, index - 1)

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

    @classmethod
    def interpolate_point(
        cls,
        line_string: "LineString"
    ) -> Callable[[float], Vec3T]:
        path_points = line_string._path_points_.value
        length_knots = line_string._length_knots_.value

        def callback(
            alpha: float
        ) -> Vec3T:
            if len(path_points) == 1:
                return path_points[0]
            index, residue = cls._integer_interpolate(alpha, length_knots)
            return SpaceUtils.lerp(path_points[index - 1], path_points[index])(residue)

        return callback

    @classmethod
    def partial(
        cls,
        line_string: "LineString"
    ) -> "Callable[[float, float], LineString]":
        path_points = line_string._path_points_.value
        length_knots = line_string._length_knots_.value

        def callback(
            start: float,
            stop: float
        ) -> LineString:
            assert start <= stop
            if len(path_points) == 1:
                new_points = [path_points[0]]
            else:
                start_index, start_residue = cls._integer_interpolate(start, length_knots, side="right")
                stop_index, stop_residue = cls._integer_interpolate(stop, length_knots, side="left")
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

        return callback

    @classmethod
    def interpolate(
        cls,
        line_string_0: "LineString",
        line_string_1: "LineString"
    ) -> "Callable[[float], LineString]":
        all_knots = np.unique(np.concatenate((line_string_0._length_knots_.value, line_string_1._length_knots_.value)))
        is_ring = False
        if line_string_0._is_ring_.value and line_string_1._is_ring_.value:
            all_knots = all_knots[:-1]
            is_ring = True

        point_interpolate_callback_0 = cls.interpolate_point(line_string_0)
        point_interpolate_callback_1 = cls.interpolate_point(line_string_1)
        point_interpolate_callbacks = [
            SpaceUtils.lerp(point_interpolate_callback_0(knot), point_interpolate_callback_1(knot))
            for knot in all_knots
        ]

        def callback(
            alpha: float
        ) -> LineString:
            return LineString(np.array([
                point_interpolate_callback(alpha)
                for point_interpolate_callback in point_interpolate_callbacks
            ]), is_ring=is_ring)

        return callback


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

    @classmethod
    def interpolate_point(
        cls,
        multi_line_string: "MultiLineString"
    ) -> Callable[[float], Vec3T]:
        line_strings = multi_line_string._line_strings_
        length_knots = multi_line_string._length_knots_.value

        def callback(
            alpha: float
        ) -> Vec3T:
            if not line_strings:
                raise ValueError("Attempting to interpolate an empty MultiLineString")
            index, residue = cls._integer_interpolate(alpha, length_knots)
            return LineString.interpolate_point(line_strings[index - 1])(residue)

        return callback

    @classmethod
    def partial(
        cls,
        multi_line_string: "MultiLineString"
    ) -> "Callable[[float, float], MultiLineString]":
        line_strings = multi_line_string._line_strings_
        length_knots = multi_line_string._length_knots_.value

        def callback(
            start: float,
            stop: float
        ) -> MultiLineString:
            assert start <= stop
            result = MultiLineString()
            if not line_strings:
                return result

            start_index, start_residue = cls._integer_interpolate(start, length_knots, side="right")
            stop_index, stop_residue = cls._integer_interpolate(stop, length_knots, side="left")
            if start_index == stop_index:
                result._line_strings_.append(
                    LineString.partial(line_strings[start_index - 1])(start_residue, stop_residue)
                )
            else:
                result._line_strings_.extend((
                    LineString.partial(line_strings[start_index - 1])(start_residue, 1.0),
                    *line_strings[start_index:stop_index - 1],
                    LineString.partial(line_strings[stop_index - 1])(0.0, stop_residue)
                ))
            return result

        return callback

    @classmethod
    def interpolate(
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
        line_string_interpolate_callbacks: list[Callable[[float], LineString]] = [
            LineString.interpolate(
                LineString.partial(line_strings_0[index_0])(start_residue_0, stop_residue_0),
                LineString.partial(line_strings_1[index_1])(start_residue_1, stop_residue_1)
            )
            for (index_0, start_residue_0, stop_residue_0), (index_1, start_residue_1, stop_residue_1) in triplet_tuple_list
        ]

        inlay_interpolate_callbacks: list[Callable[[float], Vec3sT]] = []
        for index_0, residues in enumerate(residue_list_list_0):
            point_interpolate_callback = LineString.interpolate_point(line_strings_0[index_0])
            points: Vec3sT = np.array([
                point_interpolate_callback(residue)
                for residue in residues
            ])
            if len(points) == 2:
                continue
            points_center: Vec3T = np.average(points, axis=0)
            inlay_interpolate_callbacks.append(SpaceUtils.lerp(points, points_center))

        for index_1, residues in enumerate(residue_list_list_1):
            point_interpolate_callback = LineString.interpolate_point(line_strings_1[index_1])
            points: Vec3sT = np.array([
                point_interpolate_callback(residue)
                for residue in residues
            ])
            if len(points) == 2:
                continue
            points_center: Vec3T = np.average(points, axis=0)
            inlay_interpolate_callbacks.append(SpaceUtils.lerp(points_center, points))

        def callback(
            alpha: float
        ) -> MultiLineString:
            result = MultiLineString()
            result._line_strings_.extend(
                line_string_interpolate_callback(alpha)
                for line_string_interpolate_callback in line_string_interpolate_callbacks
            )
            if has_inlay:
                result._line_strings_.extend(
                    LineString(inlay_interpolate_callback(alpha), is_ring=True)
                    for inlay_interpolate_callback in inlay_interpolate_callbacks
                )
            return result
        return callback

    @classmethod
    def concatenate(
        cls,
        *multi_line_strings: "MultiLineString"
    ) -> "Callable[[], MultiLineString]":
        result = MultiLineString()
        result._line_strings_.extend(it.chain.from_iterable(
            multi_line_string._line_strings_
            for multi_line_string in multi_line_strings
        ))

        def callback() -> MultiLineString:
            return result

        return callback
