from __future__ import annotations


from typing import Self

import numpy as np

from ...constants.constants import (
    PI,
    TAU
)
from ...constants.custom_typing import (
    NP_x2f8,
    NP_x3f8
)
from .parametric_surface import ParametricSurface


class Sphere(ParametricSurface):
    __slots__ = ()

    def __init__(
        self: Self
    ) -> None:

        def func(
            samples: NP_x2f8
        ) -> NP_x3f8:
            theta = samples[:, 0]
            phi = samples[:, 1]
            return np.column_stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))

        super().__init__(
            func=func,
            normal_func=func,
            u_range=(0.0, TAU),
            v_range=(0.0, PI),
            resolution=(32, 16)
        )
