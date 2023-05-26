import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    NP_44f8,
    NP_2f8,
    NP_f8
)
from ..lazy.lazy import Lazy


class PerspectiveCamera(Camera):
    __slots__ = ()

    @Lazy.property_array
    @classmethod
    def _projection_matrix_(
        cls,
        frame_radii: NP_2f8,
        near: NP_f8,
        far: NP_f8,
        altitude: NP_f8
    ) -> NP_44f8:
        sx, sy = altitude / frame_radii
        sz = -(far + near) / (far - near)
        tz = -2.0 * far * near / (far - near)
        return np.array((
            ( sx, 0.0,  0.0, 0.0),
            (0.0,  sy,  0.0, 0.0),
            (0.0, 0.0,   sz,  tz),
            (0.0, 0.0, -1.0, 0.0)
        ))
