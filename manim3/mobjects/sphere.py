import numpy as np

from ..mobjects.parametric_surface import ParametricSurface
from ..typing import *


__all__ = ["Sphere"]


class Sphere(ParametricSurface):
    def __init__(
        self: Self,
        theta_segments: int = 32,
        phi_segments: int = 16,
        **kwargs
    ):
        super().__init__(
            lambda theta, phi: np.array((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi))),
            (0.0, 2 * np.pi),
            (0.0, np.pi),
            resolution=(theta_segments, phi_segments),
            **kwargs
        )
