__all__ = ["PlaneGeometry"]


import numpy as np

from ..geometries.parametric_surface_geometry import ParametricSurfaceGeometry


class PlaneGeometry(ParametricSurfaceGeometry):
    __slots__ = ()

    def __new__(
        cls,
        width_segments: int = 1,
        height_segments: int = 1
    ):
        return super().__new__(
            cls,
            func=lambda x, y: np.array((x, y, 0.0)),
            normal_func=lambda x, y: np.array((0.0, 0.0, 1.0)),
            u_range=(-1.0, 1.0),
            v_range=(-1.0, 1.0),
            resolution=(width_segments, height_segments)
        )
