from .regular_polygon import RegularPolygon
#from .shapes.circle_shape import CircleShape
#from .shape_mobject import ShapeMobject


class Circle(RegularPolygon):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(n=64)
