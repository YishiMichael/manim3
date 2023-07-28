from .shapes.square_shape import SquareShape
from .shape_mobject import ShapeMobject


class Square(ShapeMobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(SquareShape())
