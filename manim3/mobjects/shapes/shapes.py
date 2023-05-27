import numpy as np

from ...constants import (
    OUT,
    PI,
    TAU
)
from ...custom_typing import (
    NP_2f8,
    NP_x2f8
)
from ...shape.shape import Shape
from ..shape_mobject import ShapeMobject


class Polyline(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        points: NP_x2f8
    ) -> None:
        super().__init__(Shape([(points, False)]))


class Point(Polyline):
    __slots__ = ()

    def __init__(
        self,
        point: NP_2f8
    ) -> None:
        super().__init__(np.array((point,)))


class Line(Polyline):
    __slots__ = ()

    def __init__(
        self,
        start_point: NP_2f8,
        stop_point: NP_2f8
    ) -> None:
        super().__init__(np.array((start_point, stop_point)))


class Arc(Polyline):
    __slots__ = ()

    def __init__(
        self,
        start_angle: float,
        sweep_angle: float
    ) -> None:
        n_segments = int(np.ceil(sweep_angle / TAU * 64.0))
        complex_coords = np.exp(1.0j * (start_angle + np.linspace(0.0, sweep_angle, n_segments + 1)))
        super().__init__(np.vstack((complex_coords.real, complex_coords.imag)).T)


class Circle(Arc):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(0.0, TAU)


class Polygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        points: NP_x2f8
    ) -> None:
        super().__init__(Shape([(points, True)]))


class RegularPolygon(Polygon):
    __slots__ = ()

    def __init__(
        self,
        n: int
    ) -> None:
        # By default, one vertex is at (1, 0).
        complex_coords = np.exp(1.0j * np.linspace(0.0, TAU, n, endpoint=False))
        super().__init__(np.vstack((complex_coords.real, complex_coords.imag)).T)


class Triangle(RegularPolygon):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(3)
        self.rotate(PI / 2.0 * OUT)


class Square(RegularPolygon):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(4)
        self.rotate(PI / 4.0 * OUT)
