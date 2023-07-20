#from typing import (
#    Callable,
#    Iterator
#)

#import numpy as np

#from ....constants.custom_typing import (
#    NP_3f8,
#    NP_x3f8,
#    NP_xf8
#)
#from ....lazy.lazy import Lazy
#from ....utils.space import SpaceUtils
#from .path_interpolant import PathInterpolant


#class LineString(PathInterpolant):
#    __slots__ = ()

#    def __init__(
#        self,
#        points: NP_x3f8
#    ) -> None:
#        assert len(points)
#        super().__init__()
#        self._points_ = points

#    @Lazy.variable_array
#    @classmethod
#    def _points_(cls) -> NP_x3f8:
#        return np.zeros((0, 3))

#    @Lazy.property_array
#    @classmethod
#    def _lengths_(
#        cls,
#        points: NP_x3f8
#    ) -> NP_xf8:
#        vectors: NP_x3f8 = np.diff(points, axis=0)
#        return SpaceUtils.norm(vectors)

#    @classmethod
#    def partial(
#        cls,
#        line_string: "LineString"
#    ) -> "Callable[[float, float], LineString]":
#        if len(line_string._points_) == 1:

#            def callback_empty(
#                start: float,
#                stop: float
#            ) -> LineString:
#                assert start < stop
#                return LineString(line_string._points_)

#            return callback_empty

#        knots = cls._lengths_to_knots(line_string._lengths_)
#        points = line_string._points_

#        def callback(
#            start: float,
#            stop: float
#        ) -> LineString:
#            assert start < stop
#            start_index, start_residue, stop_index, stop_residue = cls._partial_residues(knots, start, stop)
#            return LineString(np.array([
#                SpaceUtils.lerp(points[start_index], points[start_index + 1])(start_residue),
#                *points[start_index + 1:stop_index + 1],
#                SpaceUtils.lerp(points[stop_index], points[stop_index + 1])(stop_residue)
#            ]))

#        return callback

#    @classmethod
#    def interpolate(
#        cls,
#        line_string_0: "LineString",
#        line_string_1: "LineString"
#    ) -> "Callable[[float], LineString]":

#        def decompose_line_string(
#            line_string: LineString,
#            residues_list: list[NP_xf8]
#        ) -> Iterator[NP_3f8]:
#            points = line_string._points_
#            for index, residues in enumerate(residues_list):
#                yield from SpaceUtils.lerp(points[index], points[index + 1])(residues[:-1, None])
#            yield points[-1]

#        if len(line_string_0._points_) == 1 or len(line_string_1._points_) == 1:
#            point_interpolate_callbacks = [
#                SpaceUtils.lerp(points_0, points_1)
#                for points_0 in line_string_0._points_
#                for points_1 in line_string_1._points_
#            ]
#        else:
#            residues_list_0, residues_list_1 = cls._zip_residues_list(
#                cls._lengths_to_knots(line_string_0._lengths_),
#                cls._lengths_to_knots(line_string_1._lengths_)
#            )
#            point_interpolate_callbacks = [
#                SpaceUtils.lerp(point_0, point_1)
#                for point_0, point_1 in zip(
#                    decompose_line_string(line_string_0, residues_list_0),
#                    decompose_line_string(line_string_1, residues_list_1),
#                    strict=True
#                )
#            ]

#        def callback(
#            alpha: float
#        ) -> LineString:
#            return LineString(np.array([
#                point_interpolate_callback(alpha)
#                for point_interpolate_callback in point_interpolate_callbacks
#            ]))

#        return callback
