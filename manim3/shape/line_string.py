import itertools as it
from typing import (
    Callable,
    Iterable,
    Iterator,
    Literal
)

import numpy as np

from ..custom_typing import (
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

    @classmethod
    def _lengths_to_knots(
        cls,
        lengths: NP_xf8
    ) -> NP_xf8:
        assert len(lengths)
        lengths_cumsum = np.maximum(lengths, 1e-6).cumsum()
        return np.insert(lengths_cumsum / lengths_cumsum[-1], 0, 0.0)

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
        return int(start_index), float(start_residue), int(stop_index), float(stop_residue)

    @classmethod
    def _zip_residues_list(
        cls,
        *knots_tuple: NP_xf8
    ) -> tuple[list[NP_xf8], ...]:
        all_knots, unique_inverse = np.unique(np.concatenate(knots_tuple), return_inverse=True)
        offsets = np.insert(np.cumsum([
            len(knots) for knots in knots_tuple
        ]), 0, 0)
        return tuple(
            [
                (all_knots[start_index:stop_index + 1] - all_knots[start_index]) / (all_knots[stop_index] - all_knots[start_index])
                for start_index, stop_index in it.pairwise(unique_inverse[start_offset:stop_offset])
            ]
            for start_offset, stop_offset in it.pairwise(offsets)
        )


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
        self._points_ = points  #self._remove_duplicate_points(points, is_ring)
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
        vectors: NP_x3f8 = np.diff(path_points, axis=0)
        return SpaceUtils.norm(vectors)

    #@classmethod
    #def _remove_duplicate_points(
    #    cls,
    #    points: NP_x3f8,
    #    is_ring: bool
    #) -> NP_x3f8:
    #    nonzero_vector_indices = (np.diff(points, axis=0) != 0.0).any(axis=1).nonzero()[0]
    #    if not len(nonzero_vector_indices):
    #        return points[:1]
    #    new_points = points[nonzero_vector_indices]
    #    if not is_ring or (points[0] != points[-1]).any():
    #        new_points = np.append(new_points, points[-1:], axis=0)
    #    return new_points

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

        def callback(
            start: float,
            stop: float
        ) -> LineString:
            assert start < stop
            start_index, start_residue, stop_index, stop_residue = cls._partial_residues(knots, start, stop)
            return LineString(np.array([
                SpaceUtils.lerp(path_points[start_index], path_points[start_index + 1])(start_residue),
                *path_points[start_index + 1:stop_index + 1],
                SpaceUtils.lerp(path_points[stop_index], path_points[stop_index + 1])(stop_residue)
            ]), is_ring=False)

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
                yield from SpaceUtils.lerp(path_points[index], path_points[index + 1])(residues[:-1, None])
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
        return np.array([lengths.sum() for lengths in line_strings__lengths])

    @classmethod
    def partial(
        cls,
        multi_line_string: "MultiLineString"
    ) -> "Callable[[float, float], MultiLineString]":
        line_strings = multi_line_string._line_strings_
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
            if start >= stop:
                return MultiLineString()
            start_index, start_residue, stop_index, stop_residue = cls._partial_residues(knots, start, stop)
            if start_index == stop_index:
                return MultiLineString([
                    line_string_partial_callbacks[start_index](start_residue, stop_residue)
                ])
            return MultiLineString([
                line_string_partial_callbacks[start_index](start_residue, 1.0),
                *line_strings[start_index + 1:stop_index],
                line_string_partial_callbacks[stop_index](0.0, stop_residue)
            ])

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
            inlay_points = SpaceUtils.lerp(points[indices], points[indices + 1])(sub_residues[:, None])
            if not is_ring:
                inlay_points = np.append(inlay_points, points[-1:])
            return LineString(inlay_points, is_ring=is_ring)

        def get_inlay_line_string_placeholder(
            inlay_line_string: LineString
        ) -> LineString:
            inlay_points = inlay_line_string._points_
            inlay_points_center: NP_x3f8 = np.average(inlay_points, axis=0, keepdims=True)
            return LineString(inlay_points_center, is_ring=False)

        if len(multi_line_string_0._lengths_) == 1 and len(multi_line_string_0._lengths_) == 1:
            return [LineString.interpolate(
                multi_line_string_0._line_strings_[0],
                multi_line_string_1._line_strings_[0]
            )]

        residues_list_0, residues_list_1 = cls._zip_residues_list(
            cls._lengths_to_knots(multi_line_string_0._lengths_),
            cls._lengths_to_knots(multi_line_string_1._lengths_)
        )
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

    @classmethod
    def interpolate(
        cls,
        multi_line_string_0: "MultiLineString",
        multi_line_string_1: "MultiLineString"
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

        return callback

    @classmethod
    def concatenate(
        cls,
        *multi_line_strings: "MultiLineString"
    ) -> "Callable[[], MultiLineString]":
        result = MultiLineString(it.chain.from_iterable(
            multi_line_string._line_strings_
            for multi_line_string in multi_line_strings
        ))

        def callback() -> MultiLineString:
            return result

        return callback
