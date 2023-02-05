__all__ = ["PerspectiveCamera"]


import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    Mat4T,
    Real
)
from ..rendering.config import ConfigSingleton
from ..utils.lazy import (
    LazyData,
    lazy_basedata,
    lazy_property
)


class PerspectiveCamera(Camera):
    def __new__(
        cls,
        width: Real | None = None,
        height: Real | None = None,
        near: Real | None = None,
        far: Real | None = None,
        altitude: Real | None = None
    ):
        instance = super().__new__(cls)
        if width is not None:
            instance._width_ = LazyData(width)
        if height is not None:
            instance._height_ = LazyData(height)
        if near is not None:
            instance._near_ = LazyData(near)
        if far is not None:
            instance._far_ = LazyData(far)
        if altitude is not None:
            instance._altitude_ = LazyData(altitude)
        return instance

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
