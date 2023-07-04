from typing import (
    Callable,
    Iterator
)

import numpy as np

from ..custom_typing import (
    NP_3f8,
    NP_x3f8,
    NP_xf8
)
from ..lazy.lazy import Lazy
from ..utils.space import SpaceUtils
from .path_interpolant import PathInterpolant


class LineString(PathInterpolant):
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
        vectors: NP_x3f8 = np.diff(path_points, axis=0)
        return SpaceUtils.norm(vectors)

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
