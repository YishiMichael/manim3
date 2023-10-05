from __future__ import annotations


from typing import Self

import numpy as np

from .polygon import Polygon

#from .shapes.square_shape import SquareShape
#from .shape_mobject import ShapeMobject


class Square(Polygon):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(positions=np.array((
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0)
        )))
