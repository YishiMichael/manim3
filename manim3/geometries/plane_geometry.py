import numpy as np

from ..geometries.parametric_surface_geometry import ParametricSurfaceGeometry


class PlaneGeometry(ParametricSurfaceGeometry):
    __slots__ = ()

    def __init__(
        self,
        width_segments: int = 1,
        height_segments: int = 1
    ) -> None:
        super().__init__(
            func=lambda x, y: np.array((x, y, 0.0)),
            normal_func=lambda x, y: np.array((0.0, 0.0, 1.0)),
            u_range=(-0.5, 0.5),
            v_range=(-0.5, 0.5),
            resolution=(width_segments, height_segments)
        )
