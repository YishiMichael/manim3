import numpy as np

from geometries.parametrized_surface_geometry import ParametrizedSurfaceGeometry
from utils.typing import *


__all__ = ["PlaneGeometry"]


class PlaneGeometry(ParametrizedSurfaceGeometry):
    def __init__(self: Self, x_segments: int = 1, y_segments: int = 1):
        super().__init__(
            lambda x, y: np.array((x, y, 0.0)),
            np.linspace(-0.5, 0.5, x_segments + 1),
            np.linspace(-0.5, 0.5, y_segments + 1),
        )
