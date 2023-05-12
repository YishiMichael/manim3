import numpy as np

from ..cameras.camera import Camera
from ..config import ConfigSingleton
from ..custom_typing import Mat4T
from ..lazy.lazy import Lazy


class PerspectiveCamera(Camera):
    __slots__ = ()

    @Lazy.property_external
    @classmethod
    def _projection_matrix_(
        cls,
        width: float,
        height: float,
        near: float,
        far: float,
        altitude: float
    ) -> Mat4T:
        sx = 2.0 * altitude / width
        sy = 2.0 * altitude / height
        sz = -(far + near) / (far - near)
        tz = -2.0 * far * near / (far - near)
        return np.array((
            ( sx, 0.0,  0.0, 0.0),
            (0.0,  sy,  0.0, 0.0),
            (0.0, 0.0,   sz,  tz),
            (0.0, 0.0, -1.0, 0.0)
        ))
