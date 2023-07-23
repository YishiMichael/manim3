from .shape_mobject import ShapeMobject
from .shapes.circle_shape import CircleShape


class Circle(ShapeMobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(CircleShape())
