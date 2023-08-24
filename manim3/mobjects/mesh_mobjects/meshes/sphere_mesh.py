import numpy as np

from ....constants.constants import (
    PI,
    TAU
)
from ....constants.custom_typing import (
    NP_x2f8,
    NP_x3f8
)
from .parametric_surface_mesh import ParametricSurfaceMesh


class SphereMesh(ParametricSurfaceMesh):
    __slots__ = ()

    def __init__(self) -> None:

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
