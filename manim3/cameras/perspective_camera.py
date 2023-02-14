__all__ = ["PerspectiveCamera"]


import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    Mat4T,
    Real
)
from ..rendering.config import ConfigSingleton
from ..utils.lazy import (
    NewData,
    lazy_basedata,
    lazy_property
)


class PerspectiveCamera(Camera):
    __slots__ = ()

    def __init__(
        self,
        *,
        width: Real | None = None,
        height: Real | None = None,
        near: Real | None = None,
        far: Real | None = None,
        altitude: Real | None = None
    ):
        super().__init__()
        if width is not None:
            self._width_ = NewData(width)
        if height is not None:
            self._height_ = NewData(height)
        if near is not None:
            self._near_ = NewData(near)
        if far is not None:
            self._far_ = NewData(far)
        if altitude is not None:
            self._altitude_ = NewData(altitude)

    @lazy_basedata
    @staticmethod
    def _width_() -> Real:
        return ConfigSingleton().frame_width

    @lazy_basedata
    @staticmethod
    def _height_() -> Real:
        return ConfigSingleton().frame_height

    @lazy_basedata
    @staticmethod
    def _near_() -> Real:
        return ConfigSingleton().camera_near

    @lazy_basedata
    @staticmethod
    def _far_() -> Real:
        return ConfigSingleton().camera_far

    @lazy_basedata
    @staticmethod
    def _altitude_() -> Real:
        return ConfigSingleton().camera_altitude

    @lazy_property
    @staticmethod
    def _projection_matrix_(
        width: Real,
        height: Real,
        near: Real,
        far: Real,
        altitude: Real
    ) -> Mat4T:
        sx = 2.0 * altitude / width
        sy = 2.0 * altitude / height
        sz = -(far + near) / (far - near)
        tz = -2.0 * far * near / (far - near)
        return np.array((
            ( sx, 0.0,  0.0, 0.0),
            (0.0,  sy,  0.0, 0.0),
            (0.0, 0.0,   sz,  tz),
            (0.0, 0.0, -1.0, 0.0),
        ))
