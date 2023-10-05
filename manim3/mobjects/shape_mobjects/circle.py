from __future__ import annotations


from typing import Self

from .regular_polygon import RegularPolygon
#from .shapes.circle_shape import CircleShape
#from .shape_mobject import ShapeMobject


class Circle(RegularPolygon):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(n=64)
