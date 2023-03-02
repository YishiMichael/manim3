__all__ = [
    "Arc",
    "Circle",
    "Line",
    "Point",
    "Polygon",
    "Polyline",
    "RegularPolygon",
    "Square",
    "Triangle"
]


import numpy as np
from scipy.spatial.transform import Rotation

from ..constants import (
    OUT,
    PI,
    TAU
)
from ..custom_typing import (
    Real,
    Vec2T,
    Vec2sT
)
from ..mobjects.shape_mobject import ShapeMobject
from ..utils.shape import (
    LineString2D,
    MultiLineString2D,
    Shape
)


class Polyline(ShapeMobject):
    def __init__(
        self,
        coords: Vec2sT
    ) -> None:
        super().__init__(Shape(MultiLineString2D([LineString2D(coords)])))


class Point(Polyline):
    def __init__(
        self,
        point: Vec2T
    ) -> None:
        super().__init__(np.array((point,)))


class Line(Polyline):
    def __init__(
        self,
        start_point: Vec2T,
        stop_point: Vec2T
    ) -> None:
        super().__init__(np.array((start_point, stop_point)))


class Arc(Polyline):
    def __init__(
        self,
        start_angle: Real,
        sweep_angle: Real
    ) -> None:
        n_segments = int(np.ceil(sweep_angle / TAU * 64.0))
        complex_coords = np.exp(1.0j * (start_angle + np.linspace(0.0, sweep_angle, n_segments + 1)))
        super().__init__(np.vstack((complex_coords.real, complex_coords.imag)).T)


class Circle(Arc):
    def __init__(self) -> None:
        super().__init__(0.0, TAU)


class Polygon(Polyline):
    def __init__(
        self,
        coords: Vec2sT
    ) -> None:
        super().__init__(np.append(coords, np.array((coords[0],)), axis=0))


class RegularPolygon(Polygon):
    def __init__(
        self,
        n: int
    ) -> None:
        # By default, one vertex is at (1, 0)
        complex_coords = np.exp(1.0j * np.linspace(0.0, TAU, n, endpoint=False))
        super().__init__(np.vstack((complex_coords.real, complex_coords.imag)).T)


class Triangle(RegularPolygon):
    def __init__(self) -> None:
        super().__init__(3)
        self.rotate_about_origin(Rotation.from_rotvec(OUT * PI / 2.0))


class Square(RegularPolygon):
    def __init__(self) -> None:
        super().__init__(4)
        self.rotate_about_origin(Rotation.from_rotvec(OUT * PI / 4.0))
