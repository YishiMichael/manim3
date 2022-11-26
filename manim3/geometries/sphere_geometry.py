import numpy as np

from ..geometries.parametric_surface_geometry import ParametricSurfaceGeometry
from ..constants import PI, TAU
from ..custom_typing import *


__all__ = ["SphereGeometry"]


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
        super().__init__(
            lambda theta, phi: np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))),
            (theta_start, theta_start + theta_sweep),
            (phi_start, phi_start + phi_sweep),
            (theta_segments, phi_segments)
        )
