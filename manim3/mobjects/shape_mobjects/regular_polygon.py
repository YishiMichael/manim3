from .shape_mobject import ShapeMobject
from .shapes.regular_polygon_shape import RegularPolygonShape


class RegularPolygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        n: int
    ) -> None:
        super().__init__(RegularPolygonShape(
            n=n
        ))
