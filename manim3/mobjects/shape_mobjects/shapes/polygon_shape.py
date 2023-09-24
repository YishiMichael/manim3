import numpy as np

from ....constants.custom_typing import NP_x2f8
from .shape import Shape


class PolygonShape(Shape):
    __slots__ = ()

    def __init__(
        self,
        positions: NP_x2f8
    ) -> None:
        super().__init__(
            positions=positions,
            counts=np.array((len(positions)),)
        )
