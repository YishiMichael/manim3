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

    def __init__(
        self,
        theta_start: float = 0.0,
        theta_sweep: float = TAU,
        phi_start: float = 0.0,
        phi_sweep: float = PI,
        theta_segments: int = 32,
        phi_segments: int = 16
    ) -> None:

        def func(
            theta: NP_f8,
            phi: NP_f8
        ) -> NP_3f8:
            return np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))

        super().__init__(
            func=func,
            normal_func=func,
            u_range=(theta_start, theta_start + theta_sweep),
            v_range=(phi_start, phi_start + phi_sweep),
            resolution=(theta_segments, phi_segments)
        )
