import numpy as np

from ...constants import ORIGIN
from ...custom_typing import (
    NP_3f8,
    NP_x3f8
)
from ...shape.line_string import LineString
from ...shape.multi_line_string import MultiLineString
from ..stroke_mobject import StrokeMobject


class Polyline(StrokeMobject):
    __slots__ = ()

    def __init__(
        self,
        points: NP_x3f8
    ) -> None:
        super().__init__(MultiLineString([LineString(points, is_ring=False)]))


class Dot(Polyline):
    __slots__ = ()

    def __init__(
        self,
        point: NP_3f8 = ORIGIN
    ) -> None:
        super().__init__(np.array((point,)))


class Line(Polyline):
    __slots__ = ()

    def __init__(
        self,
        start_point: NP_3f8,
        stop_point: NP_3f8
    ) -> None:
        super().__init__(np.array((start_point, stop_point)))
