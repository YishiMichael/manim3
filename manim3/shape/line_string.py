import itertools as it
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal
)

import numpy as np

from ..custom_typing import (
    #NP_f8,
    NP_xf8,
    NP_3f8,
    NP_x3f8,
    NP_xu4
)
from ..lazy.lazy import (
    Lazy,
    LazyObject
)
from ..utils.space import SpaceUtils


class ShapeInterpolant(LazyObject):
    __slots__ = ()

    #@Lazy.property_array
    #@classmethod
    #def _lengths_(cls) -> NP_xf8:
    #    return np.zeros((0, 1))

    #@Lazy.property_array
    #@classmethod
    #def _length_(
    #    cls,
    #    lengths: NP_xf8
    #) -> NP_f8:
    #    return np.array(lengths.sum())

    #@Lazy.property_array
    #@classmethod
    #def _length_knots_(
    #    cls,
    #    lengths: NP_xf8
    #) -> NP_xf8:
    #    # `length_knots` is a collection of normalized accumulated lengths.
    #    # All entries are in the interval (0.0, 1.0) and sorted in strictly increasing order.
    #    # It has one less entry than `lengths`, and hence shall only be used when `lengths` is not empty.
    #    knots = lengths.cumsum()
    #    factor = knots[-1] if len(knots) else 1.0
    #    #if len(lengths):
    #    #    # Ensure the last entry is always precisely 1.0.
    #    #    knots /= knots[-1]
    #    return knots[:-1] / factor
    #    #unnormalized_knots = np.insert(lengths.cumsum(), 0, 0.0)
    #    #return unnormalized_knots / unnormalized_knots[-1]

    #@classmethod
    #def _get_residue(
    #    cls,
    #    target: float,
    #    array: NP_xf8,
    #    index: int
    #) -> float:
    #    return (target - array[index]) / (array[index + 1] - array[index])

    #@classmethod
    #def _integer_interpolate(
    #    cls,
    #    target: float,
    #    array: NP_xf8,
    #    *,
    #    side: Literal["left", "right"] = "right"
    #) -> tuple[int, float]:
    #    """
    #    Assumed that `array` has at least two elements and already sorted, and that
    #    `array[0] <= target <= array[-1]`.
    #    Returns `(i, (target - array[i - 1]) / (array[i] - array[i - 1]))` such that
    #    `1 <= i <= len(array) - 1` and `array[i - 1] <= target <= array[i]`.
    #    """
    #    #array = self._length_knots_.value
    #    assert (array_len := len(array)) >= 2
    #    index = int(np.searchsorted(array, target, side=side))
    #    if index == 0:
    #        index += 1
    #        alpha = 0.0
    #    elif index == array_len:
    #        index -= 1
    #        alpha = 1.0
    #    else:
    #        alpha = float((target - array[index - 1]) / (array[index] - array[index - 1]))
    #    return index, alpha


    #    #if index == 0:
    #    #    return 0, max(float(target / array[0]), 0.0)
    #    #if index == len(array):
    #    #    return len(array) - 1, 1.0
    #    #return index, float((target - array[index - 1]) / (array[index] - array[index - 1]))#index, cls._get_residue(target, array, index - 1)

    @classmethod
    def _lengths_to_knots(
        cls,
        lengths: NP_xf8
    ) -> NP_xf8:
        assert len(lengths)
        lengths_cumsum = np.maximum(lengths, 1e-6).cumsum()
        return np.insert(lengths_cumsum / lengths_cumsum[-1], 0, 0.0)

    #@classmethod
    #def _unique_knots_with_inverse(
    #    cls,
    #    *knots_tuple: NP_xf8
    #) -> tuple[NP_xf8, tuple[NP_xu4, ...]]:

        #def get_residues(
        #    knots: NP_xf8,
        #    indices: NP_xu4
        #) -> Iterator[NP_xf8]:
        #    if not len(indices):
        #        yield knots
        #        return
        #    yield knots[:indices[0]] / knots[indices[0]]
        #    for start_index, stop_index in it.pairwise(indices):
        #        yield knots[start_index + 1:stop_index] / (knots[stop_index] - knots[start_index])
        #    yield knots[indices[-1] + 1:] / (1.0 - knots[indices[-1]])

        #unique_knots, unique_inverse = np.unique(np.concatenate(knots_tuple), return_inverse=True)
        #unique_inverse = unique_inverse.astype(np.uint32)
        #offsets = np.insert(np.cumsum([
        #    len(knots) for knots in knots_tuple
        #]), 0, 0)
        #indices_iter = (
        #    unique_inverse[start_offset:stop_offset]
        #    for start_offset, stop_offset in it.pairwise(offsets)
        #)
        #return unique_knots, tuple(
        #    unique_inverse[start_offset:stop_offset]
        #    for start_offset, stop_offset in it.pairwise(offsets)
        #)
        #return tuple(
        #    [
        #        (all_knots[start_index:stop_index + 1] - all_knots[start_index]) / (all_knots[stop_index] - all_knots[start_index])
        #        for start_index, stop_index in it.pairwise(indices)
        #    ]
        #    for indices in indices_iter
        #)
        #return unique_knots, tuple(
        #    all_indices[offset:next_offset]
        #    for offset, next_offset in it.pairwise(offsets)
        #)

        #indices_list: list[NP_xu4] = []
        #offset: int = 0
        #for knots in knots_list:
        #    next_offset = offset + len(knots)
        #    indices_list.append(all_indices[offset:next_offset])
        #    offset = next_offset

        #return unique_knots, tuple(indices_list)

    #@classmethod
    #def _get_residues_list(
    #    cls,
    #    knots: NP_xf8,
    #    indices: NP_xu4
    #) -> list[NP_xf8]:
    #    return [
    #        (knots[start_index:stop_index + 1] - knots[start_index]) / (knots[stop_index] - knots[start_index])
    #        for start_index, stop_index in it.pairwise(indices)
    #    ]

    @classmethod
    def _interpolate_knots(
        cls,
        knots: NP_xf8,
        values: NP_xf8,
        *,
        side: Literal["left", "right"]
    ) -> tuple[NP_xu4, NP_xf8]:
        index = (np.searchsorted(knots, values, side=side) - 1).astype(np.uint32)
        residue = (values - knots[index]) / (knots[index + 1] - knots[index])
        return index, residue

    @classmethod
    def _partial_residues(
        cls,
        knots: NP_xf8,
        start: float,
        stop: float
    ) -> tuple[int, float, int, float]:
        start_index, start_residue = cls._interpolate_knots(knots, start * np.ones((1,)), side="right")
        stop_index, stop_residue = cls._interpolate_knots(knots, stop * np.ones((1,)), side="left")
        #start_index = int(np.searchsorted(knots, start, side="right")) - 1
        #start_residue = float((start - knots[start_index]) / (knots[start_index + 1] - knots[start_index]))
        #stop_index = int(np.searchsorted(knots, stop, side="left")) - 1
        #stop_residue = float((stop - knots[stop_index]) / (knots[stop_index + 1] - knots[stop_index]))
        return int(start_index), float(start_residue), int(stop_index), float(stop_residue)
        #all_knots, (span_indices, knot_indices) = cls._unique_knots_with_inverse(
        #    np.array((start, stop)),
        #    knots
        #)
        #residues_list = cls._get_residues_list(all_knots, knot_indices)
        #start_index = int(span_indices[0])
        #if start_index in knot_indices:
        #    start_index += 1
        #    start_residue = 0.0
        #else:
        #    start_residue = float(residues_list[start_index][0])
        #stop_index = int(span_indices[1]) + len(knots) - len(all_knots)
        #if stop_index in knot_indices:
        #    stop_index -= 1
        #    stop_residue = 1.0
        #else:
        #    stop_residue = float(residues_list[stop_index][-1])
        #return start_index, start_residue, stop_index, stop_residue

    @classmethod
    def _zip_residues_list(
        cls,
        *knots_tuple: NP_xf8
    ) -> tuple[list[NP_xf8], ...]:
        all_knots, unique_inverse = np.unique(np.concatenate(knots_tuple), return_inverse=True)
        #unique_inverse = unique_inverse.astype(np.uint32)
        offsets = np.insert(np.cumsum([
            len(knots) for knots in knots_tuple
        ]), 0, 0)
        #indices_iter = (
        #    unique_inverse[start_offset:stop_offset]
        #    for start_offset, stop_offset in it.pairwise(offsets)
        #)
        #return unique_knots, tuple(
        #    unique_inverse[start_offset:stop_offset]
        #    for start_offset, stop_offset in it.pairwise(offsets)
        #)
        return tuple(
            [
                (all_knots[start_index:stop_index + 1] - all_knots[start_index]) / (all_knots[stop_index] - all_knots[start_index])
                for start_index, stop_index in it.pairwise(unique_inverse[start_offset:stop_offset])
            ]
            for start_offset, stop_offset in it.pairwise(offsets)
        )

    #@classmethod
    #def _zip_knots(
    #    cls,
    #    *knots_lists: NP_xf8
    #) -> tuple[tuple[list[list[float]], ...], list[tuple[tuple[int, float, float], ...]]]:
    #    all_knots = np.concatenate(knots_lists)
    #    all_list_indices = np.repeat(np.arange(len(knots_lists)), [
    #        len(knots) for knots in knots_lists
    #    ])
    #    unique_knots, unique_inverse, unique_counts = np.unique(all_knots, return_inverse=True, return_counts=True)
    #    unique_inverse_argsorted = np.argsort(unique_inverse)
    #    list_indices_groups = [
    #        all_list_indices[unique_inverse_argsorted[slice(*span)]]
    #        for span in it.pairwise(np.cumsum(unique_counts))
    #    ]
    #    assert len(list_indices_groups)
    #    assert len(list_indices_groups[-1]) == len(knots_lists)
    #    knot_indices = np.zeros(len(knots_lists), dtype=np.int_)
    #    residue_list = [0.0 for _ in knots_lists]
    #    residue_list_tuple = tuple([0.0] for _ in knots_lists)
    #    residue_list_list_tuple: tuple[list[list[float]]] = tuple([] for _ in knots_lists)
    #    triplet_tuple_list: list[tuple[tuple[int, float, float], ...]] = []
    #    for knot, list_indices in zip(unique_knots[1:], list_indices_groups, strict=True):
    #        triplet_list: list[tuple[int, float, float]] = []
    #        for index in range(len(knots_lists)):
    #            if index in list_indices:
    #                stop_residue = 1.0
    #            else:
    #                stop_residue = cls._get_residue(knot, knots_lists[index], knot_indices[index])
    #            residue_list_tuple[index].append(stop_residue)
    #            triplet_list.append((knot_indices[index], residue_list[index], stop_residue))
    #            if index in list_indices:
    #                residue_list_list_tuple[index].append(residue_list_tuple[index][:])
    #                residue_list_tuple[index].clear()
    #                residue_list_tuple[index].append(0.0)
    #                next_residue = 0.0
    #                knot_indices[index] += 1
    #            else:
    #                next_residue = stop_residue
    #            residue_list[index] = next_residue
    #        triplet_tuple_list.append(tuple(triplet_list))
    #    return residue_list_list_tuple, triplet_tuple_list


class LineString(ShapeInterpolant):
    __slots__ = ()

    def __init__(
        self,
        points: NP_x3f8,
        *,
        is_ring: bool
    ) -> None:
        assert len(points)
        super().__init__()
        self._points_ = points
        self._is_ring_ = is_ring

    @Lazy.variable_array
    @classmethod
    def _points_(cls) -> NP_x3f8:
        return np.zeros((0, 3))

    @Lazy.variable_hashable
    @classmethod
    def _is_ring_(cls) -> bool:
        return False

    @Lazy.property_array
    @classmethod
    def _path_points_(
        cls,
        points: NP_x3f8,
        is_ring: bool
    ) -> NP_x3f8:
        if not is_ring:
            return points.copy()
        return np.append(points, (points[0],), axis=0)

    @Lazy.property_array
    @classmethod
    def _lengths_(
        cls,
        path_points: NP_x3f8
    ) -> NP_xf8:
        #return np.maximum(SpaceUtils.norm(np.diff(path_points, axis=0)), 1e-6)
        vectors: NP_x3f8 = np.diff(path_points, axis=0)
        return SpaceUtils.norm(vectors)

    #def remove_duplicate_points(self):
    #    # Ensure all segments have non-zero lengths.
    #    path_points = self._path_points_
    #    vectors: NP_x3f8 = np.diff(path_points, axis=0)
    #    nonzero_length_indices = SpaceUtils.norm(vectors).nonzero()[0]
    #    new_points = np.insert(path_points[nonzero_length_indices + 1], 0, path_points[0], axis=0)
    #    if self._is_ring_ and len(path_points) > 1:  # Keep one point at least.
    #        new_points = new_points[:-1]
    #    self._points_ = new_points
    #    return self

    #@classmethod
    #def interpolate_point(
    #    cls,
    #    line_string: "LineString"
    #) -> Callable[[float], NP_3f8]:
    #    path_points = line_string._path_points_
    #    length_knots = line_string._length_knots_

    #    def callback(
    #        alpha: float
    #    ) -> NP_3f8:
    #        if len(path_points) == 1:
    #            return path_points[0]
    #        index, residue = cls._integer_interpolate(alpha, length_knots)
    #        return SpaceUtils.lerp_3f8(path_points[index - 1], path_points[index])(residue)

    #    return callback

    @classmethod
    def partial(
        cls,
        line_string: "LineString"
    ) -> "Callable[[float, float], LineString]":
        if len(line_string._points_) == 1:

            def callback_empty(
                start: float,
                stop: float
            ) -> LineString:
                assert start < stop
                return LineString(line_string._points_, is_ring=False)

            return callback_empty

        knots = cls._lengths_to_knots(line_string._lengths_)
        path_points = line_string._path_points_
        #length_knots = line_string._length_knots_

        def callback(
            start: float,
            stop: float
        ) -> LineString:
            assert start < stop
            #if len(path_points) == 1:
            #    new_points = [path_points[0]]
            #else:
            start_index, start_residue, stop_index, stop_residue = cls._partial_residues(knots, start, stop)
            return LineString(np.array([
                SpaceUtils.lerp(path_points[start_index], path_points[start_index + 1])(start_residue),
                *path_points[start_index + 1:stop_index + 1],
                SpaceUtils.lerp(path_points[stop_index], path_points[stop_index + 1])(stop_residue)
            ]), is_ring=False)
            #points: list[NP_3f8] = []
            #if start_index == stop_index:
            #    points.append(
            #        SpaceUtils.lerp_3f8(path_points[start_index], path_points[start_index + 1])(start_residue)
            #        SpaceUtils.lerp_3f8(path_points[start_index], path_points[start_index + 1])(stop_residue)
            #    )
            #else:
            #    points.extend((
            #        line_string_partial_callbacks[start_index](start_residue, 1.0),
            #        *line_strings[start_index + 1:stop_index],
            #        line_string_partial_callbacks[stop_index](0.0, stop_residue)
            #    ))
            #return result
            ##start_index, start_residue = cls._integer_interpolate(start, length_knots, side="right")
            ##stop_index, stop_residue = cls._integer_interpolate(stop, length_knots, side="left")
            ##if start_index == stop_index and start_residue == stop_residue:
            ##    new_points = [
            ##        SpaceUtils.lerp_3f8(path_points[start_index - 1], path_points[start_index])(start_residue)
            ##    ]
            ##else:
            ##    new_points = [
            ##        SpaceUtils.lerp_3f8(path_points[start_index - 1], path_points[start_index])(start_residue),
            ##        *path_points[start_index:stop_index],
            ##        SpaceUtils.lerp_3f8(path_points[stop_index - 1], path_points[stop_index])(stop_residue)
            ##    ]
            #return LineString(np.array(new_points), is_ring=False)

        return callback

    @classmethod
    def interpolate(
        cls,
        line_string_0: "LineString",
        line_string_1: "LineString"
    ) -> "Callable[[float], LineString]":

        def decompose_line_string(
            line_string: LineString,
            residues_list: list[NP_xf8]
        ) -> Iterator[NP_3f8]:
            path_points = line_string._path_points_
            for index, residues in enumerate(residues_list):
                segment_partial_callback = SpaceUtils.lerp(path_points[index], path_points[index + 1])
                yield from segment_partial_callback(residues[:-1, None])
            yield path_points[-1]

        if len(line_string_0._points_) == 1 or len(line_string_1._points_) == 1:
            points_interpolate_callback = SpaceUtils.lerp(line_string_0._points_, line_string_1._points_)
            is_ring = False
            if len(line_string_0._points_) != 1:
                is_ring = line_string_0._is_ring_
            if len(line_string_1._points_) != 1:
                is_ring = line_string_1._is_ring_

            def callback_scale(
                alpha: float
            ) -> LineString:
                return LineString(points_interpolate_callback(alpha), is_ring=is_ring)

            return callback_scale

        residues_list_0, residues_list_1 = cls._zip_residues_list(
            cls._lengths_to_knots(line_string_0._lengths_),
            cls._lengths_to_knots(line_string_1._lengths_)
        )
        #all_knots, (knot_indices_0, knot_indices_1) = cls._zip_knots(
        #    multi_line_string_0._length_knots_, multi_line_string_1._length_knots_
        #)
        #segment_partial_callbacks_0 = [
        #    SpaceUtils.lerp_3f8(start_point, stop_point)
        #    for start_point, stop_point in it.pairwise(line_string_0._path_points_)
        #]
        #segment_partial_callbacks_1 = [
        #    SpaceUtils.lerp_3f8(start_point, stop_point)
        #    for start_point, stop_point in it.pairwise(line_string_1._path_points_)
        #]
        point_interpolate_callbacks: list[Callable[[float], NP_3f8]] = [
            SpaceUtils.lerp(point_0, point_1)
            for point_0, point_1 in zip(
                decompose_line_string(line_string_0, residues_list_0),
                decompose_line_string(line_string_1, residues_list_1),
                strict=True
            )
        ]
        if (is_ring := line_string_0._is_ring_ and line_string_1._is_ring_):
            point_interpolate_callbacks.pop()

        #all_knots = np.unique(np.concatenate((line_string_0._length_knots_, line_string_1._length_knots_)))
        #is_ring = False
        #if line_string_0._is_ring_ and line_string_1._is_ring_:
        #    all_knots = all_knots[:-1]
        #    is_ring = True

        #point_interpolate_callback_0 = cls.interpolate_point(line_string_0)
        #point_interpolate_callback_1 = cls.interpolate_point(line_string_1)
        #point_interpolate_callbacks = [
        #    SpaceUtils.lerp_3f8(point_interpolate_callback_0(knot), point_interpolate_callback_1(knot))
        #    for knot in all_knots
        #]

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

    @Lazy.property_array
    @classmethod
    def _lengths_(
        cls,
        line_strings__lengths: list[NP_xf8]
    ) -> NP_xf8:
        #return np.maximum(np.array([
        #    lengths.sum() for lengths in line_strings__lengths
        #]), 1e-6)
        return np.array([lengths.sum() for lengths in line_strings__lengths])

    #@classmethod
    #def interpolate_point(
    #    cls,
    #    multi_line_string: "MultiLineString"
    #) -> Callable[[float], NP_3f8]:
    #    line_strings = multi_line_string._line_strings_
    #    length_knots = multi_line_string._length_knots_

    #    def callback(
    #        alpha: float
    #    ) -> NP_3f8:
    #        if not line_strings:
    #            raise ValueError("Attempting to interpolate an empty MultiLineString")
    #        index, residue = cls._integer_interpolate(alpha, length_knots)
    #        return LineString.interpolate_point(line_strings[index - 1])(residue)

    #    return callback

    @classmethod
    def partial(
        cls,
        multi_line_string: "MultiLineString"
    ) -> "Callable[[float, float], MultiLineString]":
        line_strings = multi_line_string._line_strings_
        #length_knots = multi_line_string._length_knots_
        if not line_strings:

            def callback_empty(
                start: float,
                stop: float
            ) -> MultiLineString:
                return MultiLineString()

            return callback_empty

        knots = cls._lengths_to_knots(multi_line_string._lengths_)
        line_string_partial_callbacks = [
            LineString.partial(line_string) for line_string in line_strings
        ]

        def callback(
            start: float,
            stop: float
        ) -> MultiLineString:
            result = MultiLineString()
            if start >= stop:
                return result

            #start_index, start_residue = cls._integer_interpolate(start, length_knots, side="right")
            #stop_index, stop_residue = cls._integer_interpolate(stop, length_knots, side="left")
            start_index, start_residue, stop_index, stop_residue = cls._partial_residues(knots, start, stop)
            if start_index == stop_index:
                result._line_strings_.append(
                    line_string_partial_callbacks[start_index](start_residue, stop_residue)
                )
            else:
                result._line_strings_.extend((
                    line_string_partial_callbacks[start_index](start_residue, 1.0),
                    *line_strings[start_index + 1:stop_index],
                    line_string_partial_callbacks[stop_index](0.0, stop_residue)
                ))
            return result

        return callback

    @classmethod
    def interpolate_pieces(
        cls,
        multi_line_string_0: "MultiLineString",
        multi_line_string_1: "MultiLineString",
        *,
        has_inlay: bool
    ) -> list[Callable[[float], LineString]]:

        def decompose_multi_line_string(
            multi_line_string: MultiLineString,
            residues_list: list[NP_xf8]
        ) -> Iterator[LineString]:
            line_strings = multi_line_string._line_strings_
            for index, residues in enumerate(residues_list):
                line_string_partial_callback = LineString.partial(line_strings[index])
                for start_residue, stop_residue in it.pairwise(residues):
                    yield line_string_partial_callback(start_residue, stop_residue)

        def get_inlay_line_string(
            line_string: LineString,
            residues: NP_xf8
        ) -> LineString | None:
            points = line_string._points_
            if len(residues) == 2 or len(points) == 1:
                return None

            is_ring = line_string._is_ring_
            indices, sub_residues = cls._interpolate_knots(
                cls._lengths_to_knots(line_string._lengths_),
                residues[:-1],
                side="right"
            )
            inlay_points: NP_x3f8 = SpaceUtils.lerp(points[indices], points[indices + 1])(sub_residues)
            if not is_ring:
                inlay_points = np.append(inlay_points, points[-1:])
            return LineString(inlay_points, is_ring=is_ring)

        def get_inlay_line_string_placeholder(
            inlay_line_string: LineString
        ) -> LineString:
            inlay_points = inlay_line_string._points_
            inlay_points_center: NP_x3f8 = np.average(inlay_points, axis=0, keepdims=True)
            return LineString(inlay_points_center, is_ring=False)

        #line_strings_0 = multi_line_string_0._line_strings_
        #line_strings_1 = multi_line_string_1._line_strings_
        #if not line_strings_0 or not line_strings_1:
        #    raise ValueError("Attempting to interpolate an empty MultiLineString")

        #(residue_list_list_0, residue_list_list_1), triplet_tuple_list = cls._zip_knots(
        #    multi_line_string_0._length_knots_, multi_line_string_1._length_knots_
        #)

        if len(multi_line_string_0._lengths_) == 1 and len(multi_line_string_0._lengths_) == 1:
            return [LineString.interpolate(
                multi_line_string_0._line_strings_[0],
                multi_line_string_1._line_strings_[0]
            )]

        residues_list_0, residues_list_1 = cls._zip_residues_list(
            cls._lengths_to_knots(multi_line_string_0._lengths_),
            cls._lengths_to_knots(multi_line_string_1._lengths_)
        )
        #all_knots, (knot_indices_0, knot_indices_1) = cls._zip_knots(
        #    multi_line_string_0._length_knots_, multi_line_string_1._length_knots_
        #)
        #line_string_partial_callbacks_0 = [
        #    LineString.partial(line_string_0)
        #    for line_string_0 in multi_line_string_0._line_strings_
        #]
        #line_string_partial_callbacks_1 = [
        #    LineString.partial(line_string_1)
        #    for line_string_1 in multi_line_string_1._line_strings_
        #]
        result = [
            LineString.interpolate(line_string_0, line_string_1)
            for line_string_0, line_string_1 in zip(
                decompose_multi_line_string(multi_line_string_0, residues_list_0),
                decompose_multi_line_string(multi_line_string_1, residues_list_1),
                strict=True
            )
        ]
        if has_inlay:
            result.extend(
                LineString.interpolate(
                    inlay_line_string,
                    get_inlay_line_string_placeholder(inlay_line_string)
                )
                for line_string, residues in zip(multi_line_string_0._line_strings_, residues_list_0)
                if (inlay_line_string := get_inlay_line_string(line_string, residues)) is not None
            )
            result.extend(
                LineString.interpolate(
                    get_inlay_line_string_placeholder(inlay_line_string),
                    inlay_line_string
                )
                for line_string, residues in zip(multi_line_string_1._line_strings_, residues_list_1)
                if (inlay_line_string := get_inlay_line_string(line_string, residues)) is not None
            )
        return result
        #line_string_interpolate_callbacks: list[Callable[[float], LineString]] = [
        #    LineString.interpolate(
        #        LineString.partial(line_strings_0[index_0])(start_residue_0, stop_residue_0),
        #        LineString.partial(line_strings_1[index_1])(start_residue_1, stop_residue_1)
        #    )
        #    for (index_0, start_residue_0, stop_residue_0), (index_1, start_residue_1, stop_residue_1) in triplet_tuple_list
        #]

    #@classmethod
    #def interpolate_pieces_with_inlay(
    #    cls,
    #    multi_line_string_0: "MultiLineString",
    #    multi_line_string_1: "MultiLineString"
    #) -> list[Callable[[float], LineString]]:

    #    #knots_0 = cls._lengths_to_knots(multi_line_string_0._lengths_)
    #    #knots_1 = cls._lengths_to_knots(multi_line_string_1._lengths_)
    #    residues_list_0, residues_list_1 = cls._zip_residues_list(
    #        cls._lengths_to_knots(multi_line_string_0._lengths_),
    #        cls._lengths_to_knots(multi_line_string_1._lengths_)
    #    )
    #    result = cls.interpolate_pieces(
    #        multi_line_string_0,
    #        multi_line_string_1
    #    )
    #    return [
    #        LineString.interpolate(
    #            inlay_line_string,
    #            get_inlay_line_string_placeholder(inlay_line_string)
    #        )
    #        for line_string, residues in zip(multi_line_string_0._line_strings_, residues_list_0)
    #        if (inlay_line_string := get_inlay_line_string(line_string, residues)) is not None
    #    ] + [
    #        LineString.interpolate(
    #            get_inlay_line_string_placeholder(inlay_line_string),
    #            inlay_line_string
    #        )
    #        for line_string, residues in zip(multi_line_string_1._line_strings_, residues_list_1)
    #        if (inlay_line_string := get_inlay_line_string(line_string, residues)) is not None
    #    ]
    #    #for line_string, residues in zip(multi_line_string_0._line_strings_, residues_list_0):
    #    #    points = line_string._points_
    #    #    if len(residues) == 2 or len(points) == 1:
    #    #        continue

    #    #    #if len(points) == 1:
    #    #    #    inlay_interpolate_callbacks.append(LineString.interpolate(line_string, line_string))
    #    #    #    continue

    #    #    is_ring = line_string._is_ring_
    #    #    indices, sub_residues = cls._interpolate_knots(
    #    #        cls._lengths_to_knots(line_string._lengths_),
    #    #        residues[:-1],
    #    #        side="right"
    #    #    )
    #    #    inlay_points = SpaceUtils.lerp_x3f8(points[indices], points[indices + 1])(sub_residues)
    #    #    if not is_ring:
    #    #        inlay_points = np.append(inlay_points, points[-1:])
    #    #    inlay_points_center: NP_x3f8 = np.average(inlay_points, axis=0, keepdims=True)
    #    #    inlay_interpolate_callbacks.append(LineString.interpolate(
    #    #        LineString(inlay_points, is_ring=is_ring),
    #    #        LineString(inlay_points_center, is_ring=False)
    #    #    ))

    #    #    length_knots = line_string._length_knots_

    #    #    def callback(
    #    #        alpha: float
    #    #    ) -> NP_3f8:
    #    #        if len(path_points) == 1:
    #    #            return path_points[0]
    #    #        index, residue = cls._integer_interpolate(alpha, length_knots)
    #    #        return SpaceUtils.lerp_3f8(path_points[index - 1], path_points[index])(residue)

    #    #    #point_interpolate_callback = LineString.interpolate_point(line_strings_0[index_0])
    #    #    #points: NP_x3f8 = np.array([
    #    #    #    point_interpolate_callback(residue)
    #    #    #    for residue in residues
    #    #    #])
    #    #    #if len(points) == 2:
    #    #    #    continue
    #    #    #points_center: NP_x3f8 = np.average(points, axis=0, keepdims=True)
    #    #    #inlay_interpolate_callbacks.append(SpaceUtils.lerp_x3f8(points, points_center))
    #    #for index_1, residues in enumerate(residue_list_list_1):
    #    #    point_interpolate_callback = LineString.interpolate_point(line_strings_1[index_1])
    #    #    points: NP_x3f8 = np.array([
    #    #        point_interpolate_callback(residue)
    #    #        for residue in residues
    #    #    ])
    #    #    if len(points) == 2:
    #    #        continue
    #    #    points_center: NP_x3f8 = np.average(points, axis=0, keepdims=True)
    #    #    inlay_interpolate_callbacks.append(SpaceUtils.lerp_x3f8(points_center, points))

    @classmethod
    def interpolate(
        cls,
        multi_line_string_0: "MultiLineString",
        multi_line_string_1: "MultiLineString"
        #*,
        #has_inlay: bool = False
    ) -> "Callable[[float], MultiLineString]":
        line_string_interpolate_callbacks = cls.interpolate_pieces(
            multi_line_string_0, multi_line_string_1, has_inlay=False
        )

        def callback(
            alpha: float
        ) -> MultiLineString:
            return MultiLineString(
                line_string_interpolate_callback(alpha)
                for line_string_interpolate_callback in line_string_interpolate_callbacks
            )
            #result = MultiLineString()
            #result._line_strings_.extend(
            #    line_string_interpolate_callback(alpha)
            #    for line_string_interpolate_callback in line_string_interpolate_callbacks
            #)
            #if has_inlay:
            #    result._line_strings_.extend(
            #        LineString(inlay_interpolate_callback(alpha), is_ring=True)
            #        for inlay_interpolate_callback in inlay_interpolate_callbacks
            #    )
            #return result

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
