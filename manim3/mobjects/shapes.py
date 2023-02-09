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
    def __new__(
        cls,
        coords: Vec2sT | None = None
    ):
        if coords is None:
            coords = np.zeros((1, 2))
        return super().__new__(cls, Shape(MultiLineString2D([LineString2D(coords)])))


class Point(Polyline):
    def __new__(
        cls,
        point: Vec2T | None = None
    ):
        if point is None:
            point = np.zeros(2)
        return super().__new__(cls, np.array((point,)))


class Line(Polyline):
    def __new__(
        cls,
        start_point: Vec2T | None = None,
        stop_point: Vec2T | None = None
    ):
        if start_point is None:
            start_point = np.array((-1.0, 0.0))
        if stop_point is None:
            stop_point = np.array((1.0, 0.0))
        return super().__new__(cls, np.array((start_point, stop_point)))


class Arc(Polyline):
    def __new__(
        cls,
        start_angle: Real = 0.0,
        sweep_angle: Real = PI / 2.0
    ):
        n_segments = int(np.ceil(sweep_angle / TAU * 64.0))
        complex_coords = np.exp(1.0j * (start_angle + np.linspace(0.0, sweep_angle, n_segments + 1)))
        return super().__new__(cls, np.vstack((complex_coords.real, complex_coords.imag)).T)


class Circle(Arc):
    def __new__(cls):
        return super().__new__(cls, 0.0, TAU)


class Polygon(Polyline):
    def __new__(
        cls,
        coords: Vec2sT | None = None
    ):
        if coords is None:
            coords = np.zeros((1, 2))
        return super().__new__(cls, np.append(coords, np.array((coords[0],)), axis=0))


class RegularPolygon(Polygon):
    def __new__(
        cls,
        n: int = 3
    ):
        # By default, one vertex is at (1, 0)
        complex_coords = np.exp(1.0j * np.linspace(0.0, TAU, n, endpoint=False))
        return super().__new__(cls, np.vstack((complex_coords.real, complex_coords.imag)).T)


class Triangle(RegularPolygon):
    def __new__(cls):
        return super().__new__(cls, 3).rotate_about_origin(Rotation.from_rotvec(OUT * PI / 2.0))


class Square(RegularPolygon):
    def __new__(cls):
        return super().__new__(cls, 4).rotate_about_origin(Rotation.from_rotvec(OUT * PI / 4.0))
