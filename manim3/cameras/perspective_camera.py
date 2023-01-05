__all__ = ["PerspectiveCamera"]


import numpy as np

from ..cameras.camera import Camera
from ..constants import (
    ASPECT_RATIO,
    CAMERA_ALTITUDE,
    CAMERA_FAR,
    CAMERA_NEAR,
    DEGREES,
    FRAME_Y_RADIUS
)
from ..custom_typing import (
    Matrix44Type,
    Real
)
from ..utils.lazy import lazy_property, lazy_property_initializer_writable


class PerspectiveCamera(Camera):
    def __init__(
        self,
        fovy: Real = 2.0 * np.arctan(FRAME_Y_RADIUS / CAMERA_ALTITUDE) / DEGREES,
        aspect: Real = ASPECT_RATIO,
        near: Real = CAMERA_NEAR,
        far: Real = CAMERA_FAR
    ):
        super().__init__()
        self._fovy_ = fovy
        self._aspect_ = aspect
        self._near_ = near
        self._far_ = far

    @lazy_property_initializer_writable
    @staticmethod
    def _fovy_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _aspect_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _near_() -> Real:
        return NotImplemented

    @lazy_property_initializer_writable
    @staticmethod
    def _far_() -> Real:
        return NotImplemented

    @lazy_property
    @staticmethod
    def _projection_matrix_(
        fovy: Real,
        aspect: Real,
        near: Real,
        far: Real
    ) -> Matrix44Type:
        ymax = near * np.tan(fovy * np.pi / 360.0)
        xmax = ymax * aspect
        left = -xmax
        right = xmax
        bottom = -ymax
        top = ymax

        a = (right + left) / (right - left)
        b = (top + bottom) / (top - bottom)
        c = -(far + near) / (far - near)
        d = -2.0 * far * near / (far - near)
        e = 2.0 * near / (right - left)
        f = 2.0 * near / (top - bottom)

        return np.array((
            (  e, 0.0, 0.0,  0.0),
            (0.0,   f, 0.0,  0.0),
            (  a,   b,   c, -1.0),
            (0.0, 0.0,   d,  0.0),
        ))
