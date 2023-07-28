from .shapes.regular_polygon_shape import RegularPolygonShape
from .shape_mobject import ShapeMobject


class RegularPolygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self,
        n: int
    ) -> None:
        super().__init__(RegularPolygonShape(
            n=n
        ))
