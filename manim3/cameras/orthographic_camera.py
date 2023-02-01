__all__ = ["OrthographicCamera"]


import numpy as np

from ..cameras.camera import Camera
from ..config import Config
from ..custom_typing import (
    Mat4T,
    Real
)
from ..utils.lazy import (
    lazy_property,
    lazy_property_writable
)


class OrthographicCamera(Camera):
    def __init__(
        self,
        width: Real = Config.frame_width,
        height: Real = Config.frame_height,
        near: Real = Config.camera_near,
        far: Real = Config.camera_far
    ):
        super().__init__()
        self._width_ = width
        self._height_ = height
        self._near_ = near
        self._far_ = far

    @lazy_property_writable
    @staticmethod
    def _width_() -> Real:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _height_() -> Real:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _near_() -> Real:
        return NotImplemented

    @lazy_property_writable
    @staticmethod
    def _far_() -> Real:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _projection_matrix_(
        width: Real,
        height: Real,
        near: Real,
        far: Real
    ) -> Mat4T:
        sx = 2.0 / width
        sy = 2.0 / height
        sz = -2.0 / (far - near)
        tz = -(far + near) / (far - near)
        return np.array((
            ( sx, 0.0, 0.0, 0.0),
            (0.0,  sy, 0.0, 0.0),
            (0.0, 0.0,  sz,  tz),
            (0.0, 0.0, 0.0, 1.0)
        ))
