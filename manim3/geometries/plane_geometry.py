__all__ = ["PlaneGeometry"]


import numpy as np

from ..geometries.parametric_surface_geometry import ParametricSurfaceGeometry


class PlaneGeometry(ParametricSurfaceGeometry):
    def __init__(
        self,
        width_segments: int = 1,
        height_segments: int = 1
    ):
        super().__init__(
            lambda x, y: np.array((x, y, 0.0)),
            (-1.0, 1.0),
            (-1.0, 1.0),
            (width_segments, height_segments)
        )
