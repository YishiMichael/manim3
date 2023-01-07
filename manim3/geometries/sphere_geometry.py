__all__ = ["SphereGeometry"]


import numpy as np

from ..constants import (
    PI,
    TAU
)
from ..custom_typing import (
    Real,
    Vector3Type
)
from ..geometries.parametric_surface_geometry import ParametricSurfaceGeometry


class SphereGeometry(ParametricSurfaceGeometry):
    def __init__(
        self,
        theta_start: Real = 0.0,
        theta_sweep: Real = TAU,
        phi_start: Real = 0.0,
        phi_sweep: Real = PI,
        theta_segments: int = 32,
        phi_segments: int = 16
    ):
        def func(theta: float, phi: float) -> Vector3Type:
            return np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)))

        super().__init__(
            func=func,
            normal_func=func,
            u_range=(theta_start, theta_start + theta_sweep),
            v_range=(phi_start, phi_start + phi_sweep),
            resolution=(theta_segments, phi_segments)
        )
