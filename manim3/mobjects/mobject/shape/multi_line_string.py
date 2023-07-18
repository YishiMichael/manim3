import itertools as it
from typing import (
    Callable,
    Iterable,
    Iterator
)

import numpy as np

from ....constants.custom_typing import (
    NP_x3f8,
    NP_xf8
)
from ....lazy.lazy import Lazy
from ....utils.space import SpaceUtils
from .line_string import LineString
from .path_interpolant import PathInterpolant


class MultiLineString(PathInterpolant):
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
