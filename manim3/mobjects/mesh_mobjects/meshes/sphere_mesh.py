import numpy as np

from ....constants.constants import (
    PI,
    TAU
)
from ....constants.custom_typing import (
    NP_3f8,
    NP_f8
)
from .parametric_surface_mesh import ParametricSurfaceMesh


class SphereMesh(ParametricSurfaceMesh):
    __slots__ = ()

    def __init__(self) -> None:

        def func(
            theta: NP_f8,
            phi: NP_f8
        ) -> NP_3f8:
            return np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))

        super().__init__(
            func=func,
            normal_func=func,
            u_range=(0.0, TAU),
            v_range=(0.0, PI),
            resolution=(32, 16)
        )
