import numpy as np

from .polygon_shape import PolygonShape


class SquareShape(PolygonShape):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(positions=np.array((
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0)
        )))
