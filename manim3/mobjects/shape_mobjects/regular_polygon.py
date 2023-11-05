from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.constants import TAU
from .polygon import Polygon


class RegularPolygon(Polygon):
    __slots__ = ()

    def __init__(
        self: Self,
        n: int
    ) -> None:
        # By default, one of positions is at (1, 0).
        complex_positions = np.exp(1.0j * np.linspace(0.0, TAU, n, endpoint=False))
        super().__init__(
            positions=np.vstack((complex_positions.real, complex_positions.imag)).T
        )
