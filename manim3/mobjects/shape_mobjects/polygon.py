from __future__ import annotations


from typing import Self

import numpy as np

from ...animatables.shape import Shape
from ...constants.custom_typing import NP_x2f8
from .shape_mobject import ShapeMobject


class Polygon(ShapeMobject):
    __slots__ = ()

    def __init__(
        self: Self,
        coordinates: NP_x2f8
    ) -> None:
        super().__init__(Shape(
            coordinates=coordinates,
            counts=np.array((len(coordinates),))
        ))
