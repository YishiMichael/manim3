from .regular_polygon_shape import RegularPolygonShape


class CircleShape(RegularPolygonShape):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(n=64)
