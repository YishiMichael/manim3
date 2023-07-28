from .shapes.circle_shape import CircleShape
from .shape_mobject import ShapeMobject


class Circle(ShapeMobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(CircleShape())
