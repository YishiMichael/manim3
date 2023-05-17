import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    Mat4T,
    Vec2T
)
from ..lazy.lazy import Lazy


class OrthographicCamera(Camera):
    __slots__ = ()

    @Lazy.property_external
    @classmethod
    def _projection_matrix_(
        cls,
        frame_radii: Vec2T,
        near: float,
        far: float
    ) -> Mat4T:
        sx, sy = 1.0 / frame_radii
        sz = -2.0 / (far - near)
        tz = -(far + near) / (far - near)
        return np.array((
            ( sx, 0.0, 0.0, 0.0),
            (0.0,  sy, 0.0, 0.0),
            (0.0, 0.0,  sz,  tz),
            (0.0, 0.0, 0.0, 1.0)
        ))
