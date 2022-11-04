import numpy as np

from geometries.parametrized_surface_geometry import ParametrizedSurfaceGeometry
from utils.typing import *


__all__ = ["SphereGeometry"]


class SphereGeometry(ParametrizedSurfaceGeometry):
    def __init__(self: Self, theta_segments: int = 32, phi_segments: int = 16):
        super().__init__(
            lambda theta, phi: np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))),
            np.linspace(0.0, 2 * np.pi, theta_segments + 1),
            np.linspace(0.0, np.pi, phi_segments + 1),
        )
