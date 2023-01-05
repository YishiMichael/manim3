__all__ = ["PerspectiveCamera"]


import numpy as np
import pyrr

from ..cameras.camera import Camera
from ..constants import ASPECT_RATIO, FRAME_Y_RADIUS
from ..constants import CAMERA_ALTITUDE, CAMERA_FAR, CAMERA_NEAR
from ..constants import DEGREES
from ..utils.lazy import lazy_property, lazy_property_initializer_writable
from ..custom_typing import *


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
        return pyrr.matrix44.create_perspective_projection(
            fovy,
            aspect,
            near,
            far
        )
