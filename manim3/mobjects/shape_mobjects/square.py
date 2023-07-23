from .shape_mobject import ShapeMobject
from .shapes.square_shape import SquareShape


class Square(ShapeMobject):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(SquareShape())
