from __future__ import annotations


from typing import Self

import numpy as np

from .polygon import Polygon


class Square(Polygon):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:
        super().__init__(coordinates=np.array((
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
            (1.0, -1.0)
        )))
