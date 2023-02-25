__all__ = ["PerspectiveCamera"]


import numpy as np

from ..cameras.camera import Camera
from ..custom_typing import (
    Mat4T,
    Real
)
from ..rendering.config import ConfigSingleton
from ..utils.lazy import (
    LazyWrapper,
    lazy_object_raw,
    lazy_property_raw
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
    ) -> None:
        super().__init__()
        if width is not None:
            self._width_ = LazyWrapper(width)
        if height is not None:
            self._height_ = LazyWrapper(height)
        if near is not None:
            self._near_ = LazyWrapper(near)
        if far is not None:
            self._far_ = LazyWrapper(far)
        if altitude is not None:
            self._altitude_ = LazyWrapper(altitude)

    @lazy_object_raw
    @staticmethod
    def _width_() -> Real:
        return ConfigSingleton().frame_width

    @lazy_object_raw
    @staticmethod
    def _height_() -> Real:
        return ConfigSingleton().frame_height

    @lazy_object_raw
    @staticmethod
    def _near_() -> Real:
        return ConfigSingleton().camera_near

    @lazy_object_raw
    @staticmethod
    def _far_() -> Real:
        return ConfigSingleton().camera_far

    @lazy_object_raw
    @staticmethod
    def _altitude_() -> Real:
        return ConfigSingleton().camera_altitude

    @lazy_property_raw
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
